"""Gated inference for the MC4R binding-affinity model.

What this model may be used for is decided by its own measured performance, not
by its availability. Three findings from :mod:`atlas.ml.train` set the gates:

1. **It must not be applied to peptides.** Trained on small-molecule-like
   ligands and tested on peptide-like ones, every model scored *worse than
   predicting the training mean* (best MAE 1.46 against a 1.31 baseline, with
   Spearman near zero). That is measured negative transfer, so peptide-like
   input is refused. This matters directly here: PT-141, melanotan I and
   melanotan II are peptides acting at melanocortin receptors, and they fall on
   the wrong side of this boundary.

2. **Its honest accuracy is the scaffold-split figure.** A random split shares
   51% of its test scaffolds with training and reports MAE 0.478 / R-squared
   0.699. The scaffold split shares none and reports MAE 0.681 / R-squared
   0.297. The second number is the one quoted.

3. **Conformal intervals lose their guarantee off-distribution.** Coverage is
   0.92 under a random split, 0.79 under a scaffold split and as low as 0.25
   under modality holdout. Intervals are therefore reported with the coverage
   actually measured on the relevant split, never with the nominal level alone.

The deployed model is the Tanimoto nearest-neighbour regressor: the simplest
family tried, and the best on the scaffold split. Gradient boosting won on the
random split and lost out-of-scaffold, which is worth stating plainly --
additional model complexity did not survive an honest split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from rdkit import Chem

from atlas.core.evidence import AbstentionReason, EvidenceClass
from atlas.ml.dataset import load_dataset
from atlas.ml.features import (
    Modality,
    applicability,
    classify_modality,
    featurize,
    morgan_matrix,
)
from atlas.ml.splits import scaffold_split
from atlas.ml.train import CONFORMAL_ALPHA, MODELS, SEED, evaluate

logger = logging.getLogger(__name__)

__all__ = ["PredictionResult", "MC4RPredictor", "get_predictor", "MODEL_CARD"]

#: Model family deployed, chosen on scaffold-split performance.
DEPLOYED_MODEL = "fingerprint_knn"

#: Below this maximum Tanimoto similarity to the training set, a molecule is
#: treated as outside the domain the model was fitted on.
APPLICABILITY_FLOOR = 0.30

MODEL_CARD: dict[str, Any] = {
    "name": "mc4r_pki_knn",
    "endpoint": "MC4R binding affinity (pKi, -log10 M)",
    "target": "CHEMBL259 — Melanocortin receptor 4 (Homo sapiens)",
    "model_family": "Tanimoto nearest-neighbour regression on Morgan fingerprints",
    "quality_tier": "internally_validated",
    "quality_tier_meaning": (
        "Evaluated on held-out data from the same source with a scaffold split. "
        "It has not been tested against an independent external dataset or "
        "prospectively, so it is not externally validated."
    ),
    "intended_domain": (
        "Small-molecule-like ligands of MC4R resembling the ChEMBL medicinal "
        "chemistry literature for this target."
    ),
    "out_of_domain": (
        "Peptide-like molecules, and any molecule with maximum Tanimoto "
        "similarity below 0.30 to the training set."
    ),
    "headline_performance": {
        "split": "Bemis-Murcko scaffold split (0 shared scaffolds)",
        "mae_log_units": 0.681,
        "rmse_log_units": 0.856,
        "r2": 0.297,
        "spearman": 0.590,
        "conformal_coverage_at_90_nominal": 0.794,
    },
    "random_split_comparison": {
        "mae_log_units": 0.478,
        "r2": 0.699,
        "fraction_test_scaffolds_seen_in_training": 0.511,
        "note": (
            "Reported only to show how much a random split flatters this model. "
            "It is not the figure to quote."
        ),
    },
    "negative_transfer": {
        "experiment": "trained on small-molecule-like, tested on peptide-like",
        "best_model_mae": 1.461,
        "baseline_mean_mae": 1.305,
        "verdict": (
            "Every model scored worse than predicting the training mean. The "
            "model must not be applied to peptides."
        ),
    },
    "limitations": [
        "Ki from binding assays only; IC50 and EC50 were excluded, so this does "
        "not describe functional potency.",
        "Human MC4R only. Says nothing about selectivity over MC1R, MC3R or MC5R.",
        "Binding affinity is not efficacy, exposure or safety.",
        "Trained on published medicinal chemistry, which is biased toward "
        "molecules someone chose to synthesise and report.",
    ],
    "license": "Derived from ChEMBL, CC BY-SA 3.0.",
}


@dataclass
class PredictionResult:
    """A gated prediction, or a refusal carrying its reason."""

    smiles: str
    predicted_pki: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    interval_coverage_measured: float | None = None
    interval_nominal: float = 1.0 - CONFORMAL_ALPHA
    max_tanimoto_to_training: float | None = None
    modality: str | None = None
    nearest_neighbours: list[dict[str, Any]] = field(default_factory=list)
    evidence_class: str = EvidenceClass.EVALUATED_INFERENCE.value
    abstained: bool = False
    abstention_reason: str | None = None
    abstention_explanation: str | None = None
    what_would_resolve_it: list[str] = field(default_factory=list)
    model_version: str | None = None
    dataset_version: str | None = None

    @property
    def predicted_ki_nm(self) -> float | None:
        """Affinity in nM, which is how a chemist reads it."""
        if self.predicted_pki is None:
            return None
        return float(10 ** (9.0 - self.predicted_pki))


class MC4RPredictor:
    """Fits on load and answers gated queries.

    The model is re-fitted from the versioned dataset rather than unpickled, so
    there is no stored artefact that can drift from the code that made it.
    Nearest-neighbour fitting is effectively instantaneous at this scale.
    """

    def __init__(self) -> None:
        bundle = load_dataset()
        fs = featurize(bundle.records)

        # Train only on the population the model is licensed for.
        small_mask = fs.mask(Modality.SMALL_MOLECULE_LIKE)
        self._fs_small = fs.subset(np.where(small_mask)[0])
        self.dataset_version = bundle.version

        spec = next(s for s in MODELS if s.name == DEPLOYED_MODEL)
        self.spec = spec

        # Hold out a calibration set for conformal intervals, exactly as the
        # evaluation did, so the quoted half-width corresponds to a real
        # procedure rather than to the full-data fit.
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(self._fs_small))
        n_cal = max(50, int(0.15 * len(idx)))
        self._cal_idx, self._fit_idx = idx[:n_cal], idx[n_cal:]

        X = self._fs_small.fingerprints.astype(bool)
        y = self._fs_small.y
        self.model = spec.build()
        self.model.fit(X[self._fit_idx], y[self._fit_idx])

        cal_resid = np.abs(self.model.predict(X[self._cal_idx]) - y[self._cal_idx])
        n = len(cal_resid)
        level = min(1.0, np.ceil((n + 1) * (1 - CONFORMAL_ALPHA)) / n)
        self.half_width = float(np.quantile(cal_resid, level))

        self._train_fps = self._fs_small.fingerprints[self._fit_idx]
        self._train_ids = [self._fs_small.ids[i] for i in self._fit_idx]
        self._train_y = y[self._fit_idx]

        logger.info(
            "MC4R predictor fitted on %d small-molecule-like ligands "
            "(conformal half-width %.3f log units)",
            len(self._fit_idx), self.half_width,
        )

    # -- gates --------------------------------------------------------------
    def _abstain(
        self, smiles: str, reason: AbstentionReason, explanation: str,
        resolvers: list[str], **extra: Any,
    ) -> PredictionResult:
        return PredictionResult(
            smiles=smiles,
            abstained=True,
            evidence_class=EvidenceClass.INSUFFICIENT.value,
            abstention_reason=reason.value,
            abstention_explanation=explanation,
            what_would_resolve_it=resolvers,
            model_version=MODEL_CARD["name"],
            dataset_version=self.dataset_version,
            **extra,
        )

    def predict(self, smiles: str, *, n_neighbours: int = 3) -> PredictionResult:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return self._abstain(
                smiles, AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY,
                "The structure could not be parsed, so no features can be "
                "computed for it.",
                ["Supply a valid SMILES string or a database identifier."],
            )

        modality = classify_modality(mol)
        if modality is Modality.PEPTIDE_LIKE:
            nt = MODEL_CARD["negative_transfer"]
            return self._abstain(
                smiles, AbstentionReason.OUT_OF_APPLICABILITY_DOMAIN,
                "This molecule is peptide-like, and the model is measurably "
                "unreliable on peptides. Trained on small-molecule-like "
                f"ligands and tested on peptide-like ones it reached MAE "
                f"{nt['best_model_mae']:.2f} log units against "
                f"{nt['baseline_mean_mae']:.2f} for simply predicting the "
                "training mean, with rank correlation near zero. A number from "
                "this model would be worse than no number.",
                [
                    "A model fitted on peptide-ligand affinity data for this "
                    "target, evaluated on held-out peptides.",
                    "Direct measurement of binding affinity for this molecule.",
                ],
                modality=modality.value,
            )

        fp = morgan_matrix([mol])

        sim = float(applicability(self._train_fps, fp)[0])
        if sim < APPLICABILITY_FLOOR:
            return self._abstain(
                smiles, AbstentionReason.OUT_OF_APPLICABILITY_DOMAIN,
                f"The closest training molecule has Tanimoto similarity "
                f"{sim:.2f}, below the {APPLICABILITY_FLOOR:.2f} floor. The "
                "model has not seen chemistry resembling this molecule, so a "
                "prediction would be an extrapolation with no measured error "
                "rate.",
                [
                    "Measured affinity for this chemotype, which would extend "
                    "the training domain.",
                ],
                modality=modality.value, max_tanimoto_to_training=sim,
            )

        pred = float(self.model.predict(fp.astype(bool))[0])

        # Surface the neighbours driving the prediction: for a similarity model
        # this is the whole explanation.
        sims = applicability(fp, self._train_fps)
        order = np.argsort(-sims)[:n_neighbours]
        neighbours = [
            {
                "molecule_chembl_id": self._train_ids[i],
                "tanimoto": float(sims[i]),
                "measured_pki": float(self._train_y[i]),
            }
            for i in order
        ]

        return PredictionResult(
            smiles=smiles,
            predicted_pki=pred,
            interval_low=pred - self.half_width,
            interval_high=pred + self.half_width,
            interval_coverage_measured=(
                MODEL_CARD["headline_performance"]["conformal_coverage_at_90_nominal"]
            ),
            max_tanimoto_to_training=sim,
            modality=modality.value,
            nearest_neighbours=neighbours,
            evidence_class=EvidenceClass.EVALUATED_INFERENCE.value,
            model_version=MODEL_CARD["name"],
            dataset_version=self.dataset_version,
        )


@lru_cache(maxsize=1)
def get_predictor() -> MC4RPredictor:
    return MC4RPredictor()
