"""Affinity lookup and gated inference, measurement first.

The request behind this module was to serve the registry's melanocortin
peptides rather than refuse them. Investigating the data produced a better
answer than a model: **PT-141, melanotan I and melanotan II all have exact
structural matches in ChEMBL with measured MC4R binding affinities.** Returning
a prediction for a molecule whose affinity has been measured, and which sits in
the model's own training set, would be worse than useless -- it would replace
direct experimental evidence (class A) with a weaker inference (class C).

So the order is: measured value, then model, then abstention.

For peptides with no measurement, the honest answer is still a refusal, and the
numbers say why. A peptide-specific model was built and evaluated
(:mod:`atlas.ml.peptide_train`). On a Taylor-Butina cluster split -- the only
split that genuinely separates analogue series for this series -- the best
model reached MAE 1.208 against a mean-predictor baseline of 1.324: an 8.8%
improvement, R-squared 0.073, and conformal coverage 0.574 against a 0.90
nominal level. Error did not fall with similarity to training (MAE ~1.22 across
every band above 0.6 Tanimoto), so there is no sub-domain where the model
becomes trustworthy. That does not support point predictions.

A second negative result is worth recording: **the peptide-specific features did
not help.** Residue composition, macrocyclisation and terminal state were built
and validated at 94.8% against HELM sequences, and on every split the best model
still used plain Morgan fingerprints. The peptide feature block was beaten by
structure fingerprints, and on the cluster split ridge-on-peptide-features was
worse than predicting the mean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rdkit import Chem, RDLogger

from atlas.core.evidence import AbstentionReason, EvidenceClass
from atlas.ml.dataset import ActivityRecord, load_dataset
from atlas.ml.features import Modality, classify_modality

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

__all__ = [
    "AffinityAnswer",
    "PeptideAffinityService",
    "get_affinity_service",
    "PEPTIDE_MODEL_FINDINGS",
]

#: What the peptide modelling experiment established, and the decision it drove.
PEPTIDE_MODEL_FINDINGS: dict[str, Any] = {
    "endpoint": "MC4R binding affinity (pKi) for peptide-like ligands",
    "n_peptides": 628,
    "deployed": False,
    "decision": (
        "No point predictions are served for peptides without a measured value. "
        "The model beats a mean predictor by 8.8% on a cluster split, explains "
        "7% of variance, and its 90% intervals cover 57% of cases. That is not "
        "a usable predictor, and reporting a number from it would misrepresent "
        "the evidence."
    ),
    "cluster_split": {
        "best_model": "ridge on Morgan fingerprints",
        "mae": 1.208,
        "baseline_mae": 1.324,
        "improvement_over_baseline": 0.088,
        "r2": 0.073,
        "spearman": 0.411,
        "conformal_coverage_at_90_nominal": 0.574,
    },
    "optimistic_splits_for_contrast": {
        "random_mae": 0.584,
        "scaffold_mae": 0.62,
        "note": (
            "Both are misleading here. The scaffold split reports zero shared "
            "scaffolds while leaving 100% of test peptides with a training "
            "neighbour above 0.7 Tanimoto, because changing one residue in a "
            "macrocycle changes the whole Murcko scaffold."
        ),
    },
    "improvement_over_small_molecule_model": {
        "small_molecule_model_mae_on_peptides": 1.675,
        "peptide_model_mae": 1.208,
        "note": (
            "Modality-specific modelling is the right direction -- the peptide "
            "model is clearly better than applying the small-molecule model -- "
            "but better is not the same as good enough to deploy."
        ),
    },
    "peptide_features_did_not_help": {
        "validated_against_helm": 0.948,
        "note": (
            "Residue-composition and topology features were built and validated "
            "at 94.8% against HELM sequences, yet Morgan fingerprints beat them "
            "on every split. On the cluster split, ridge on peptide features "
            "was worse than predicting the mean."
        ),
    },
    "what_would_change_this": [
        "Affinity data for peptide chemotypes beyond the handful of analogue "
        "series ChEMBL holds for this target.",
        "True sequences for more than the 28% of entries carrying HELM, which "
        "would permit order-aware rather than composition-only features.",
        "A pretrained peptide sequence model, if sequences become available at "
        "scale.",
    ],
}


@dataclass
class AffinityAnswer:
    """A measured value, a model prediction, or a refusal -- always labelled."""

    query_smiles: str
    source_kind: str  # "measurement" | "prediction" | "abstention"
    evidence_class: str = EvidenceClass.INSUFFICIENT.value
    modality: str | None = None

    # measurement
    pki: float | None = None
    pki_min: float | None = None
    pki_max: float | None = None
    is_discordant: bool = False
    ki_nm: float | None = None
    molecule_chembl_id: str | None = None
    n_replicates: int | None = None
    replicate_spread: float | None = None
    document_ids: list[str] = field(default_factory=list)
    match_kind: str | None = None  # "exact_structure" | "same_skeleton"

    # abstention
    abstained: bool = False
    abstention_reason: str | None = None
    abstention_explanation: str | None = None
    what_would_resolve_it: list[str] = field(default_factory=list)

    dataset_version: str | None = None
    target: str = "CHEMBL259 — Melanocortin receptor 4 (Homo sapiens)"

    @property
    def is_measured(self) -> bool:
        return self.source_kind == "measurement"


def _inchikey(mol: Chem.Mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


class PeptideAffinityService:
    """Answers affinity queries from measurement where possible."""

    def __init__(self) -> None:
        bundle = load_dataset()
        self.dataset_version = bundle.version
        self._by_key: dict[str, ActivityRecord] = {}
        self._by_skeleton: dict[str, ActivityRecord] = {}

        for rec in bundle.records:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            key = _inchikey(mol)
            if not key:
                continue
            self._by_key.setdefault(key, rec)
            # The first InChIKey block is the skeleton: same connectivity,
            # possibly different stereochemistry or protonation state.
            self._by_skeleton.setdefault(key.split("-")[0], rec)

        logger.info(
            "affinity service indexed %d measured molecules", len(self._by_key)
        )

    def _measurement(self, rec: ActivityRecord, smiles: str, match: str,
                     modality: Modality) -> AffinityAnswer:
        return AffinityAnswer(
            query_smiles=smiles,
            source_kind="measurement",
            evidence_class=EvidenceClass.DIRECT_EXPERIMENTAL.value,
            modality=modality.value,
            pki=rec.pki,
            pki_min=rec.pki_min,
            pki_max=rec.pki_max,
            is_discordant=rec.is_discordant,
            ki_nm=float(10 ** (9.0 - rec.pki)),
            molecule_chembl_id=rec.molecule_chembl_id,
            n_replicates=rec.n_replicates,
            replicate_spread=rec.replicate_spread,
            document_ids=list(rec.document_ids),
            match_kind=match,
            dataset_version=self.dataset_version,
        )

    def lookup(self, smiles: str) -> AffinityAnswer:
        """Measured affinity first; a structured refusal otherwise."""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return AffinityAnswer(
                query_smiles=smiles, source_kind="abstention", abstained=True,
                abstention_reason=AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY.value,
                abstention_explanation=(
                    "The structure could not be parsed, so it can neither be "
                    "matched against measured data nor featurised."
                ),
                what_would_resolve_it=["A valid SMILES string or a database identifier."],
                dataset_version=self.dataset_version,
            )

        modality = classify_modality(mol)
        key = _inchikey(mol)

        if key and key in self._by_key:
            return self._measurement(self._by_key[key], smiles, "exact_structure", modality)

        if key and key.split("-")[0] in self._by_skeleton:
            answer = self._measurement(
                self._by_skeleton[key.split("-")[0]], smiles, "same_skeleton", modality
            )
            # Same connectivity but different stereochemistry or salt form is
            # evidence about a related molecule, not about this one.
            answer.evidence_class = EvidenceClass.EXPLORATORY_HYPOTHESIS.value
            return answer

        if modality is Modality.PEPTIDE_LIKE:
            f = PEPTIDE_MODEL_FINDINGS["cluster_split"]
            return AffinityAnswer(
                query_smiles=smiles, source_kind="abstention", abstained=True,
                modality=modality.value,
                abstention_reason=AbstentionReason.ENDPOINT_NOT_VALIDATED.value,
                abstention_explanation=(
                    "No measured affinity exists for this peptide in the dataset, "
                    "and the peptide model is not good enough to stand in for one. "
                    f"On a cluster split it reached MAE {f['mae']:.3f} against a "
                    f"mean-predictor baseline of {f['baseline_mae']:.3f} "
                    f"({f['improvement_over_baseline']:.1%} better), explained "
                    f"{f['r2']:.1%} of variance, and its 90% intervals covered "
                    f"{f['conformal_coverage_at_90_nominal']:.0%} of cases. "
                    "Error did not fall with similarity to the training set, so "
                    "there is no sub-domain where it becomes reliable."
                ),
                what_would_resolve_it=list(
                    PEPTIDE_MODEL_FINDINGS["what_would_change_this"]
                ),
                dataset_version=self.dataset_version,
            )

        return AffinityAnswer(
            query_smiles=smiles, source_kind="abstention", abstained=True,
            modality=modality.value,
            abstention_reason=AbstentionReason.NO_SOURCE_EVIDENCE.value,
            abstention_explanation=(
                "No measured MC4R affinity for this molecule in the dataset. "
                "The small-molecule model may still apply; query it directly."
            ),
            what_would_resolve_it=[
                "A measured binding affinity for this molecule against MC4R.",
            ],
            dataset_version=self.dataset_version,
        )


@lru_cache(maxsize=1)
def get_affinity_service() -> PeptideAffinityService:
    return PeptideAffinityService()
