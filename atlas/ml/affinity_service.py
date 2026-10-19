"""The single entry point for "what is this compound's MC4R affinity?".

Four gates run in order, and each can only weaken the answer:

1. **Measurement.** If the affinity has been measured, report it (class A).
2. **Identity and applicability.** Unparseable or far from the training domain
   stops here.
3. **Target relevance.** A number is published only when there is reason to
   think the compound engages this target at all. Asked about Pinealon, the
   earlier model returned pKi 5.45 purely because a fingerprint could be
   computed; 1 of its 18 publications mentions melanocortin. That is now a
   class C screening hypothesis with no number attached.
4. **Calibrated prediction.** Intervals come from a scaffold-disjoint
   calibration block, which measured 90.9% coverage against a 90% nominal
   level, in place of the in-distribution calibration that measured 83.8%.

Nothing here produces a knowledge-graph edge except a measurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from atlas.core.evidence import AbstentionReason, EvidenceClass
from atlas.ml.calibration import (
    ConformalCalibrator,
    SIMILARITY_BANDS,
    band_of,
    four_way_scaffold_split,
)
from atlas.ml.dataset import load_dataset
from atlas.ml.features import (
    Modality,
    applicability,
    classify_modality,
    featurize,
    morgan_matrix,
)
from atlas.ml.target_relevance import (
    MC4R_TARGET,
    TargetEvidence,
    TargetRelevance,
    TargetRelevanceAssessor,
)
from atlas.ml.train import SEED

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

__all__ = ["AffinityResult", "AffinityService", "get_affinity_service", "CALIBRATION_CARD"]

ALPHA = 0.10
APPLICABILITY_FLOOR = 0.30
ACTIVE_PKI_FLOOR = 6.0

#: Measured calibration behaviour, from ``atlas.ml.calibrate_experiment``.
CALIBRATION_CARD: dict[str, Any] = {
    "method": "split conformal, scaffold-disjoint calibration block",
    "nominal_coverage": 0.90,
    "empirical_coverage": 0.909,
    "previous_method": "calibration drawn from the training pool",
    "previous_empirical_coverage": 0.838,
    "median_interval_width_log": 2.636,
    "data_roles": "train / select / calibrate / test, all scaffold-disjoint",
    "is_adequately_calibrated": True,
    "coverage_by_similarity": {
        "0.0–0.4": {"n": 6, "coverage": 0.500},
        "0.4–0.55": {"n": 22, "coverage": 0.909},
        "0.55–0.7": {"n": 79, "coverage": 0.924},
        "0.7–1.0": {"n": 289, "coverage": 0.913},
    },
    "coverage_by_modality": {
        "peptide_like": {"n": 134, "coverage": 0.925},
        "small_molecule_like": {"n": 262, "coverage": 0.901},
    },
    "limitations": [
        "Conformal coverage holds under exchangeability of calibration and "
        "test data. Under chemotype or modality shift that assumption fails, "
        "and nominal coverage is not guaranteed.",
        "The lowest similarity band held only 6 test molecules, so its 0.50 "
        "coverage is an observation on six points, not an estimate.",
        "Correct coverage required intervals about 0.5 log units wider. A "
        "median width of 2.64 log units spans roughly a 430-fold range in Ki: "
        "the intervals are honest, and they are wide because the model is "
        "imprecise on unseen chemotypes.",
        "Validation is internal. The model has not been tested against an "
        "independent external dataset or prospectively.",
    ],
}


@dataclass
class AffinityResult:
    """One answer, carrying the evidence that justifies its form."""

    query_smiles: str
    answer_kind: str                       # measurement | prediction | abstention
    evidence_class: str
    target: str = MC4R_TARGET["pref_name"]
    modality: str | None = None

    # measurement
    pki: float | None = None
    pki_min: float | None = None
    pki_max: float | None = None
    ki_nm: float | None = None
    is_discordant: bool = False
    molecule_chembl_id: str | None = None
    n_replicates: int | None = None
    n_documents: int | None = None

    # prediction
    predicted_pki: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    interval_nominal: float = 1.0 - ALPHA
    interval_empirical_coverage: float | None = None
    similarity_band: str | None = None
    max_similarity: float | None = None
    nearest_neighbours: list[dict[str, Any]] = field(default_factory=list)

    # relevance + abstention
    target_relevance: str | None = None
    target_relevance_label: str | None = None
    target_rationale: str | None = None
    target_evidence: dict[str, Any] = field(default_factory=dict)
    may_create_graph_edge: bool = False
    abstention_reason: str | None = None
    abstention_explanation: str | None = None
    what_would_resolve_it: list[str] = field(default_factory=list)

    calibration: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    dataset_version: str | None = None

    @property
    def abstained(self) -> bool:
        return self.answer_kind == "abstention"


class AffinityService:
    """Measurement lookup, target-relevance gating and calibrated inference."""

    def __init__(self, assessor: TargetRelevanceAssessor | None = None) -> None:
        bundle = load_dataset()
        self.dataset_version = bundle.version
        self.assessor = assessor or TargetRelevanceAssessor()

        fs = featurize(bundle.records)
        self._fs = fs
        self._records = {r.molecule_chembl_id: r for r in bundle.records}

        # Measurement index.
        self._by_key: dict[str, str] = {}
        for i, smi in enumerate(fs.smiles):
            mol = fs.mols[i]
            try:
                self._by_key.setdefault(Chem.MolToInchiKey(mol), fs.ids[i])
            except Exception:
                continue

        # Four disjoint roles. The model is fitted on train+select (selection
        # already happened offline); calibration uses its own scaffold block so
        # the quantile is estimated under the same shift it is applied to.
        split = four_way_scaffold_split(fs, seed=SEED)
        self._split = split
        X = fs.fingerprints.astype(bool)
        fit_idx = np.concatenate([split.train_idx, split.select_idx])
        self._fit_idx = fit_idx

        self.model = KNeighborsRegressor(
            n_neighbors=5, metric="jaccard", weights="distance")
        self.model.fit(X[fit_idx], fs.y[fit_idx])

        resid = np.abs(self.model.predict(X[split.calib_idx]) - fs.y[split.calib_idx])
        sim_cal = applicability(fs.fingerprints[fit_idx],
                                fs.fingerprints[split.calib_idx])
        self.calibrator = ConformalCalibrator(alpha=ALPHA, mondrian=True).fit(
            resid, sim_cal)

        # Documented actives, used for family-level target evidence.
        active = fs.y[fit_idx] >= ACTIVE_PKI_FLOOR
        self._active_idx = fit_idx[active]
        self._train_fps = fs.fingerprints[fit_idx]

        logger.info(
            "affinity service: %d measured molecules, fitted on %d, calibrated "
            "on %d scaffold-disjoint molecules",
            len(self._by_key), len(fit_idx), len(split.calib_idx),
        )

    # -- helpers -----------------------------------------------------------
    def _abstain(self, smiles: str, reason: AbstentionReason, explanation: str,
                 resolvers: Sequence[str], **extra: Any) -> AffinityResult:
        return AffinityResult(
            query_smiles=smiles, answer_kind="abstention",
            evidence_class=EvidenceClass.INSUFFICIENT.value,
            abstention_reason=reason.value, abstention_explanation=explanation,
            what_would_resolve_it=list(resolvers),
            dataset_version=self.dataset_version, **extra,
        )

    def _neighbours(self, fp: np.ndarray, k: int = 3) -> list[dict[str, Any]]:
        sims = applicability(fp, self._train_fps)
        order = np.argsort(-sims)[:k]
        out = []
        for j in order:
            idx = self._fit_idx[j]
            out.append({
                "molecule_chembl_id": self._fs.ids[idx],
                "tanimoto": float(sims[j]),
                "measured_pki": float(self._fs.y[idx]),
            })
        return out

    # -- main --------------------------------------------------------------
    def answer(self, smiles: str, *, names: Sequence[str] = (),
               check_literature: bool = True) -> AffinityResult:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return self._abstain(
                smiles, AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY,
                "The structure could not be parsed, so it can neither be matched "
                "against measured data nor featurised.",
                ["A valid SMILES string or a database identifier."],
            )

        modality = classify_modality(mol)

        # --- gate 1: a measurement outranks any inference.
        try:
            key = Chem.MolToInchiKey(mol)
        except Exception:
            key = None
        if key and key in self._by_key:
            rec = self._records[self._by_key[key]]
            evidence = self.assessor.assess(
                names=names, measured_pki=rec.pki,
                measurement_source=rec.molecule_chembl_id,
            )
            return AffinityResult(
                query_smiles=smiles, answer_kind="measurement",
                evidence_class=EvidenceClass.DIRECT_EXPERIMENTAL.value,
                modality=modality.value,
                pki=rec.pki, pki_min=rec.pki_min, pki_max=rec.pki_max,
                ki_nm=float(10 ** (9.0 - rec.pki)),
                is_discordant=rec.is_discordant,
                molecule_chembl_id=rec.molecule_chembl_id,
                n_replicates=rec.n_replicates,
                n_documents=len(rec.document_ids),
                target_relevance=evidence.relevance.value,
                target_relevance_label=evidence.relevance.label,
                target_rationale=evidence.rationale,
                may_create_graph_edge=True,
                dataset_version=self.dataset_version,
                limitations=(
                    ["Reported values disagree across studies; the median "
                     "summarises a contested measurement."]
                    if rec.is_discordant else []
                ),
            )

        # --- gate 2: applicability domain.
        fp = morgan_matrix([mol])
        similarity = float(applicability(self._train_fps, fp)[0])
        in_domain = similarity >= APPLICABILITY_FLOOR

        active_sims = applicability(self._fs.fingerprints[self._active_idx], fp)
        max_active_sim = float(active_sims[0]) if len(active_sims) else 0.0
        sims_to_actives = applicability(fp, self._fs.fingerprints[self._active_idx])
        nearest_active = int(np.argmax(sims_to_actives)) if len(sims_to_actives) else None
        nearest_active_id = (
            self._fs.ids[self._active_idx[nearest_active]]
            if nearest_active is not None else None
        )
        nearest_active_pki = (
            float(self._fs.y[self._active_idx[nearest_active]])
            if nearest_active is not None else None
        )

        # --- gate 3: target relevance.
        evidence = self.assessor.assess(
            names=names,
            max_similarity_to_actives=max_active_sim,
            nearest_active_id=nearest_active_id,
            nearest_active_pki=nearest_active_pki,
            in_applicability_domain=in_domain,
            check_literature=check_literature,
        )
        ev_payload = {
            "max_similarity_to_documented_actives": evidence.max_similarity_to_actives,
            "nearest_documented_active": evidence.nearest_active_id,
            "nearest_documented_active_pki": evidence.nearest_active_pki,
            "total_publications": evidence.total_publications,
            "target_publications": evidence.target_publications,
            "target_publication_fraction": evidence.target_publication_fraction,
            "literature_supports_target": evidence.literature_supports_target,
        }
        shared = dict(
            modality=modality.value,
            max_similarity=similarity,
            similarity_band=band_of(similarity),
            target_relevance=evidence.relevance.value,
            target_relevance_label=evidence.relevance.label,
            target_rationale=evidence.rationale,
            target_evidence=ev_payload,
        )

        if not evidence.may_publish_affinity:
            reason = (
                AbstentionReason.OUT_OF_APPLICABILITY_DOMAIN if not in_domain
                else AbstentionReason.ENDPOINT_NOT_VALIDATED
            )
            return self._abstain(
                smiles, reason, evidence.rationale,
                evidence.caveats or [
                    "A measured binding affinity for this molecule at this target.",
                ],
                **shared,
            )

        # --- gate 4: a calibrated prediction.
        pred = float(self.model.predict(fp.astype(bool))[0])
        half = self.calibrator.half_width(similarity)
        band = band_of(similarity)
        band_cov = CALIBRATION_CARD["coverage_by_similarity"].get(band, {})

        return AffinityResult(
            query_smiles=smiles, answer_kind="prediction",
            evidence_class=EvidenceClass.EVALUATED_INFERENCE.value,
            predicted_pki=pred,
            interval_low=pred - half, interval_high=pred + half,
            interval_empirical_coverage=band_cov.get("coverage"),
            nearest_neighbours=self._neighbours(fp),
            may_create_graph_edge=False,
            calibration=CALIBRATION_CARD,
            limitations=list(CALIBRATION_CARD["limitations"]),
            dataset_version=self.dataset_version,
            **shared,
        )


@lru_cache(maxsize=1)
def get_affinity_service() -> AffinityService:
    return AffinityService()
