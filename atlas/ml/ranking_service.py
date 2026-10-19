"""Serve the one claim this model has evidence for: an ordering.

The affinity service returns a number where it may and refuses where it may
not. This service answers a narrower question that survives the refusal: given
two peptides of the same series, which binds the receptor more tightly?

Everything it does is conditional on three checks, all made before an answer
is formed:

1. **The ranking gate passed.** Measured separately from the affinity gate,
   with a bound taken over molecules rather than pairs. If it did not pass, no
   ordering is offered at all.
2. **Both molecules are answerable.** The ranking is composed from the
   affinity service's own answers, so it inherits measurement-first lookup,
   target-relevance gating and abstention. If either molecule is refused, the
   comparison is refused.
3. **The two molecules are of one series.** The gate's accuracy was measured
   within series, so the claim is bounded to within series, and the condition
   is checked from the two queries by Tanimoto similarity -- no label needed.

A ranking of two **measured** molecules is a comparison of measurements and is
labelled as such. A ranking involving a prediction is an ordering with an
accuracy attached. Neither is ever a numeric affinity estimate, and this
service returns no predicted pKi even when it has one internally.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RankingAnswer", "RankingRefusal", "RankingService", "get_ranking_service",
]


class RankingRefusal(str, enum.Enum):
    GATE_NOT_PASSED = "ranking_gate_not_passed"
    UNRESOLVED_MOLECULE = "unresolved_molecular_identity"
    MOLECULE_REFUSED = "one_molecule_has_no_usable_answer"
    NOT_SAME_SERIES = "molecules_are_not_of_one_series"
    IDENTICAL_MOLECULES = "the_two_structures_are_the_same"


@dataclass
class RankingAnswer:
    """An ordering, or a refusal, with the evidence that decides which."""

    answer_kind: str                       # ranking | abstention
    basis: str | None = None               # measurement | prediction | mixed
    tighter: str | None = None             # "a" | "b" | "indistinguishable"
    confidence: float | None = None        # measured accuracy of this claim
    confidence_interval: list[float] | None = None
    pair_similarity: float | None = None
    margin_log_units: float | None = None  # |difference| behind the ordering
    margin_is_meaningful: bool | None = None

    a: dict[str, Any] = field(default_factory=dict)
    b: dict[str, Any] = field(default_factory=dict)

    refusal_reason: str | None = None
    refusal_explanation: str | None = None
    what_would_resolve_it: list[str] = field(default_factory=list)

    limitations: list[str] = field(default_factory=list)
    gate: dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.answer_kind == "abstention"


class RankingService:
    """Compose two gated affinity answers into an ordering, or refuse."""

    def __init__(self) -> None:
        from atlas.ml.affinity_service import get_affinity_service
        from atlas.ml.ranking_gate import RANKING_GATE, RANKING_PATH

        self.affinity = get_affinity_service()
        self.criteria = dict(RANKING_GATE)
        self.report: dict[str, Any] = {}
        if RANKING_PATH.exists():
            import json
            self.report = json.loads(RANKING_PATH.read_text())
        self.within = (self.report.get("evaluations", {})
                       .get("within_series_held_out_molecules", {}))
        self.gate_passed = bool(
            self.report.get("verdict", {}).get("within_series_passes"))

    # --------------------------------------------------------- internals ----

    def _gate_summary(self) -> dict[str, Any]:
        c = self.within.get("clustered", {})
        return {
            "within_series_ranking_passes": self.gate_passed,
            "accuracy": self.within.get("pair_weighted_accuracy"),
            "accuracy_interval": [c.get("lower"), c.get("upper")],
            "n_held_out_molecules": c.get("n_groups"),
            "meaningful_delta_log_units": self.criteria["meaningful_delta_log_units"],
            "pair_similarity_min": self.criteria["pair_similarity_min"],
            "affinity_gate": (
                "unchanged and still failing; this service publishes no "
                "numeric affinity estimate"
            ),
        }

    def _refuse(self, reason: RankingRefusal, explanation: str,
                resolve: Sequence[str] = ()) -> RankingAnswer:
        return RankingAnswer(
            answer_kind="abstention", refusal_reason=reason.value,
            refusal_explanation=explanation,
            what_would_resolve_it=list(resolve),
            gate=self._gate_summary(),
        )

    def _similarity(self, smiles_a: str, smiles_b: str) -> float | None:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit import DataStructs

        mols = [Chem.MolFromSmiles(s) for s in (smiles_a, smiles_b)]
        if any(m is None for m in mols):
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [gen.GetFingerprint(m) for m in mols]
        return float(DataStructs.TanimotoSimilarity(*fps))

    @staticmethod
    def _value_of(result: Any) -> tuple[float | None, str | None]:
        """The internal number used for ordering, and what kind it is.

        Deliberately private to this service: it is used to decide an order and
        is never returned.
        """
        if result.answer_kind == "measurement":
            return result.pki, "measurement"
        if result.answer_kind == "prediction":
            return result.predicted_pki, "prediction"
        return None, None

    # ------------------------------------------------------------ public ----

    def rank(self, smiles_a: str, smiles_b: str, *,
             names_a: Sequence[str] = (), names_b: Sequence[str] = ()) -> RankingAnswer:
        if not self.gate_passed:
            return self._refuse(
                RankingRefusal.GATE_NOT_PASSED,
                "The within-series ranking gate has not been passed on the "
                "current dataset, so no ordering is offered.",
                ["Re-running atlas.ml.ranking_gate after the model or data change."],
            )

        similarity = self._similarity(smiles_a, smiles_b)
        if similarity is None:
            return self._refuse(
                RankingRefusal.UNRESOLVED_MOLECULE,
                "At least one structure could not be parsed, so neither "
                "membership of a series nor an ordering can be established.",
                ["A valid SMILES string for both molecules."],
            )
        if similarity >= 0.999:
            return self._refuse(
                RankingRefusal.IDENTICAL_MOLECULES,
                "The two structures are the same molecule, so there is nothing "
                "to order.",
                ["Two different structures."],
            )

        floor = self.criteria["pair_similarity_min"]
        if similarity < floor:
            return self._refuse(
                RankingRefusal.NOT_SAME_SERIES,
                f"The two molecules share only {similarity:.2f} Tanimoto "
                f"similarity, below the {floor} that defines one analogue "
                "series. Ranking accuracy was measured within series; across "
                "series the same ranker fails its criteria, so no ordering is "
                "offered here.",
                ["Two analogues of the same series.",
                 "A measured affinity for either molecule."],
            )

        res_a = self.affinity.answer(smiles_a, names=tuple(names_a))
        res_b = self.affinity.answer(smiles_b, names=tuple(names_b))
        val_a, kind_a = self._value_of(res_a)
        val_b, kind_b = self._value_of(res_b)

        if val_a is None or val_b is None:
            which = "the first" if val_a is None else "the second"
            refused = res_a if val_a is None else res_b
            return self._refuse(
                RankingRefusal.MOLECULE_REFUSED,
                f"No usable answer exists for {which} molecule, so the pair "
                f"cannot be ordered. The affinity service refused it: "
                f"{refused.abstention_explanation or refused.abstention_reason}",
                list(refused.what_would_resolve_it)
                or ["A measured affinity for that molecule."],
            )

        margin = float(val_a - val_b)
        meaningful = abs(margin) >= self.criteria["meaningful_delta_log_units"]
        basis = ("measurement" if kind_a == kind_b == "measurement"
                 else "prediction" if kind_a == kind_b == "prediction" else "mixed")
        c = self.within.get("clustered", {})

        answer = RankingAnswer(
            answer_kind="ranking", basis=basis,
            tighter=("a" if margin > 0 else "b") if meaningful else "indistinguishable",
            confidence=(None if basis == "measurement"
                        else self.within.get("pair_weighted_accuracy")),
            confidence_interval=(None if basis == "measurement"
                                 else [c.get("lower"), c.get("upper")]),
            pair_similarity=similarity,
            margin_log_units=abs(margin),
            margin_is_meaningful=meaningful,
            a={"answer_kind": res_a.answer_kind, "basis": kind_a,
               "evidence_class": res_a.evidence_class},
            b={"answer_kind": res_b.answer_kind, "basis": kind_b,
               "evidence_class": res_b.evidence_class},
            gate=self._gate_summary(),
        )

        if basis == "measurement":
            answer.limitations.append(
                "Both molecules have measured affinities, so this is a "
                "comparison of measurements rather than a model ranking. No "
                "model accuracy applies to it."
            )
        else:
            answer.limitations.append(
                "An ordering, not an affinity estimate. The accuracy quoted is "
                f"the measured within-series figure over "
                f"{c.get('n_groups', '?')} held-out molecules and applies to "
                "pairs differing by at least "
                f"{self.criteria['meaningful_delta_log_units']} log units."
            )
        if not meaningful:
            answer.limitations.append(
                f"The internal difference behind this pair is "
                f"{abs(margin):.2f} log units, below the threshold at which "
                "the ranking was validated, so the two are reported as "
                "indistinguishable rather than ordered."
            )
        if basis == "mixed":
            answer.limitations.append(
                "One molecule is measured and the other predicted, so the "
                "comparison mixes evidence classes and is weaker than either "
                "a measurement comparison or a within-series model ranking."
            )
        answer.limitations.append(
            "No numeric affinity is returned for either molecule. The "
            "affinity deployment gate is unchanged and still fails."
        )
        return answer


@lru_cache(maxsize=1)
def get_ranking_service() -> RankingService:
    return RankingService()
