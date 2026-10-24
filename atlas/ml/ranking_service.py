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
4. **The representation can tell the two apart.** Every feature block behind
   this service is chirality-blind, so two stereoisomers produce identical
   rows and any ordering between them is decided by nothing. The MC4R dataset
   contains 25 such groups, and their measured affinities differ by a median
   1.47 and as much as 3.60 log units -- a pair the earlier version of this
   service called "the same molecule", which was false. They are now refused
   as what they are.

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
    #: Different molecules with the same constitution. Every representation
    #: this service is built on is chirality-blind, so the two receive
    #: identical feature rows and any ordering would be an artefact.
    STEREOISOMERS = "stereoisomers_the_model_cannot_distinguish"
    #: Different constitutions that the folded fingerprint still cannot
    #: separate. Rarer than the stereo case and handled the same way.
    INDISTINGUISHABLE = "structures_the_representation_cannot_separate"


@dataclass
class RankingAnswer:
    """An ordering, or a refusal, with the evidence that decides which."""

    answer_kind: str                       # ranking | abstention
    basis: str | None = None               # measurement | prediction | mixed
    tighter: str | None = None             # "a" | "b" | "indistinguishable"
    confidence: float | None = None        # measured accuracy of this claim
    confidence_interval: list[float] | None = None
    pair_similarity: float | None = None
    similarity_band: str | None = None
    #: True when the pair sits above 0.8 mutual similarity, where measured
    #: accuracy is *lower* than the pooled figure. A near-identical pair with a
    #: real affinity gap is an activity cliff.
    is_activity_cliff: bool = False
    pooled_accuracy: float | None = None   # the headline, for reference only
    #: Whether this pair's own similarity band clears the gate criterion on its
    #: own evidence. The gate passed on the pooled served domain, but the
    #: >0.8 band's lower bound is 0.797 against a 0.80 requirement, so a cliff
    #: pair sits in a band that does not independently meet the bar.
    band_meets_gate: bool | None = None
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
        self.bands = self.within.get("accuracy_by_similarity_band", {})
        self.cliff_threshold = float(
            self.report.get("cliff_band", {}).get("threshold", 0.8))
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

    def _band_for(self, similarity: float) -> dict[str, Any] | None:
        """The measured band this pair falls in, matching the gate's edges.

        The pair's own band is quoted rather than the pooled figure, because
        the pooled figure would flatter a cliff pair and understate an easy
        one.
        """
        edges = ((0.8, ">0.8"), (0.6, "0.6-0.8"), (0.4, "0.4-0.6"), (0.0, "0-0.4"))
        for lo, label in edges:
            if similarity >= lo:
                band = self.bands.get(label)
                return {**band, "label": label} if band else None
        return None

    @staticmethod
    def _structural_relation(smiles_a: str, smiles_b: str) -> str:
        """``identical``, ``stereoisomers`` or ``distinct``.

        Decided on canonical SMILES rather than on a fingerprint. A folded
        Morgan fingerprint built without chirality gives a stereoisomeric pair
        a Tanimoto of exactly 1.0, which an earlier version of this service
        read as "the same molecule" -- a false statement about two compounds
        whose measured affinities differ by up to 3.60 log units.
        """
        from rdkit import Chem

        mols = [Chem.MolFromSmiles(s) for s in (smiles_a, smiles_b)]
        if any(m is None for m in mols):
            return "distinct"
        if Chem.MolToSmiles(mols[0]) == Chem.MolToSmiles(mols[1]):
            return "identical"
        flat = []
        for m in mols:
            copy = Chem.Mol(m)
            Chem.RemoveStereochemistry(copy)
            flat.append(Chem.MolToSmiles(copy))
        return "stereoisomers" if flat[0] == flat[1] else "distinct"

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
        # The relation is computed here and used twice: the identity case is
        # decided now, and the stereoisomer case only once it is known whether
        # a model estimate is involved at all.
        relation = self._structural_relation(smiles_a, smiles_b)
        if relation == "identical":
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

        # Whether the model can tell the two structures apart only matters
        # when the model is what produced a value. Two molecules with measured
        # affinities are compared as measurements, and a pair of
        # stereoisomers is then a perfectly orderable pair -- which is the
        # case for the widest-spread pair in the MC4R set, both of whose
        # members are measured.
        if basis != "measurement":
            if relation == "stereoisomers":
                return self._refuse(
                    RankingRefusal.STEREOISOMERS,
                    "These are two different molecules with the same "
                    "constitution, differing only in stereochemistry, and at "
                    "least one side of the comparison is a model estimate. "
                    "Every representation behind this ranking is "
                    "chirality-blind, so the two produce identical feature "
                    "rows and an ordering between them would be decided by "
                    "nothing. The difference is not small in this dataset: "
                    "among the 25 stereoisomer groups in the MC4R set the "
                    "measured affinities differ by a median 1.47 and as much "
                    "as 3.60 log units, so the ordering is refused rather "
                    "than guessed.",
                    ["A measured affinity for both molecules, which would make "
                     "this a comparison of measurements rather than of "
                     "estimates.",
                     "A stereochemistry-aware model, which ATLAS has measured "
                     "and does not yet have: adding backbone configuration to "
                     "the features left cross-series error unchanged or worse."],
                )
            if similarity >= 0.999:
                return self._refuse(
                    RankingRefusal.INDISTINGUISHABLE,
                    f"The two structures are different but share a Tanimoto "
                    f"similarity of {similarity:.3f} on the fingerprint this "
                    "service uses, so it cannot separate them, and an "
                    "ordering resting on a model estimate would be an "
                    "artefact of the representation rather than a statement "
                    "about the molecules.",
                    ["A measured affinity for both molecules."],
                )

        c = self.within.get("clustered", {})
        band = self._band_for(similarity)
        is_cliff = similarity >= self.cliff_threshold
        required = self.criteria["within_series_accuracy_lower_bound_min"]
        band_ok = (None if band is None or band.get("lower") is None
                   else bool(band["lower"] >= required))

        answer = RankingAnswer(
            answer_kind="ranking", basis=basis,
            tighter=("a" if margin > 0 else "b") if meaningful else "indistinguishable",
            # The band's own figure, not the pooled one.
            confidence=(None if basis == "measurement"
                        else (band or {}).get("accuracy",
                                              self.within.get("pair_weighted_accuracy"))),
            confidence_interval=(None if basis == "measurement" else (
                [band.get("lower"), band.get("upper")] if band
                else [c.get("lower"), c.get("upper")])),
            pooled_accuracy=(None if basis == "measurement"
                             else self.within.get("pair_weighted_accuracy")),
            similarity_band=(band or {}).get("label"),
            band_meets_gate=(None if basis == "measurement" else band_ok),
            is_activity_cliff=bool(is_cliff and basis != "measurement"),
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
            where = (f"the {band['label']} similarity band, measured over "
                     f"{band.get('n_groups', '?')} held-out molecules and "
                     f"{band.get('n', '?')} pairs" if band
                     else f"{c.get('n_groups', '?')} held-out molecules")
            answer.limitations.append(
                "An ordering, not an affinity estimate. The accuracy quoted is "
                f"for {where}, and applies to pairs differing by at least "
                f"{self.criteria['meaningful_delta_log_units']} log units."
            )
        if answer.is_activity_cliff:
            pooled = answer.pooled_accuracy
            answer.limitations.insert(0, (
                f"These two structures are {similarity:.2f} similar, above "
                f"{self.cliff_threshold}, which is the band where this ranker "
                "is **least** reliable: a near-identical pair with a real "
                "affinity gap is an activity cliff. Measured accuracy here is "
                + (f"{answer.confidence:.1%}" if answer.confidence else "lower")
                + (f" against {pooled:.1%} pooled across all served pairs"
                   if pooled else "")
                + ". Treat this ordering with more caution than a less similar "
                "pair, not less."
            ))
            if band_ok is False:
                answer.limitations.insert(1, (
                    f"On its own evidence this band does not clear the gate: "
                    f"its 95% lower bound is {band['lower']:.1%} against the "
                    f"{required:.0%} required. The gate passed on the pooled "
                    "served domain, which this band is part of and drags down. "
                    "The ordering is still offered, with that stated, because "
                    "concealing it would make the pooled figure read as though "
                    "it applied uniformly."
                ))
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
