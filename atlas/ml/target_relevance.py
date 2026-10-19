"""Target relevance: is there reason to think this compound engages this target?

A model will return an affinity for any molecule whose fingerprint it can
compute. That number is a statement about chemical similarity, not about
biology. Asked for Pinealon -- a pineal tripeptide with 18 indexed publications,
one of which mentions melanocortin at all -- the MC4R model returned pKi 5.45.
Nothing about that number implies Pinealon engages MC4R in a cell.

This module separates the four cases the platform must keep apart:

``A_DOCUMENTED_ENGAGEMENT``
    A measured activity for this exact molecule at this target exists.

``B_VALIDATED_DOMAIN_PREDICTION``
    No measurement, but the molecule sits inside the model's validated domain
    *and* independent evidence ties it to this target family.

``C_EXPLORATORY_SCREENING_HYPOTHESIS``
    Inside the applicability domain, but target relevance is not established.
    A structural resemblance to known ligands is a reason to run an assay, not
    a reason to report an affinity. No number is published for this class.

``D_UNSUPPORTED``
    Outside the applicability domain, or no evidence of any kind.

Only class A may become an interaction edge in the knowledge graph. A
prediction, at any class, never does.
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from atlas.analysis.feasibility import build_query
from atlas.sources.pubmed import PubMedClient

logger = logging.getLogger(__name__)

__all__ = [
    "literature_names",
    "TargetRelevance",
    "TargetEvidence",
    "TargetRelevanceAssessor",
    "MC4R_TARGET",
]

#: Target definition, including the literature terms that identify it.
MC4R_TARGET = {
    "target_chembl_id": "CHEMBL259",
    "pref_name": "Melanocortin receptor 4",
    "organism": "Homo sapiens",
    # Deliberately MC4R-specific. A broader "melanocortin" term matches the
    # whole family -- ACTH, alpha-MSH, MC1R through MC5R -- and fires for any
    # melanocortin-adjacent peptide. Semax, an ACTH(4-10) analogue, matched 160
    # of its 214 publications on the broad term while having no established
    # MC4R pharmacology. Target relevance has to mean *this* target.
    "literature_terms": (
        "MC4R[tiab] OR \"MC4 receptor\"[tiab] OR \"melanocortin 4\"[tiab] "
        "OR \"melanocortin-4\"[tiab] OR \"receptor, melanocortin, type 4\"[mh]"
    ),
}

#: Minimum similarity to a *measured active* ligand for family-level evidence.
FAMILY_SIMILARITY_FLOOR = 0.70

#: A measured ligand counts as active above this pKi.
ACTIVE_PKI_FLOOR = 6.0

#: Literature thresholds. Both must hold: a single incidental paper is not
#: evidence of engagement, and a handful of papers out of thousands is noise.
MIN_TARGET_PAPERS = 5
MIN_TARGET_FRACTION = 0.05

#: Short alphanumeric aliases are ambiguous in a literature search and are
#: dropped from it. "MT-2" retrieves metallothionein-2 papers and inflated
#: melanotan II's apparent publication count from 199 to 2728, which deflates
#: the target fraction and hides a real association. The primary name is always
#: kept, however short.
_AMBIGUOUS_ALIAS = re.compile(r"^[A-Za-z]{1,3}[\s-]?\d{0,3}$")


def literature_names(names: Sequence[str]) -> list[str]:
    """Drop ambiguous short aliases, always keeping the primary name."""
    if not names:
        return []
    primary, rest = names[0], names[1:]
    return [primary] + [
        n for n in rest if n and not _AMBIGUOUS_ALIAS.match(n.strip())
    ]


class TargetRelevance(str, enum.Enum):
    A_DOCUMENTED_ENGAGEMENT = "A_documented_experimental_engagement"
    B_VALIDATED_DOMAIN_PREDICTION = "B_prediction_within_validated_domain"
    C_EXPLORATORY_SCREENING_HYPOTHESIS = "C_exploratory_screening_hypothesis"
    D_UNSUPPORTED = "D_unsupported_target_relationship"

    @property
    def label(self) -> str:
        return {
            TargetRelevance.A_DOCUMENTED_ENGAGEMENT:
                "Documented experimental engagement",
            TargetRelevance.B_VALIDATED_DOMAIN_PREDICTION:
                "Prediction within a validated domain",
            TargetRelevance.C_EXPLORATORY_SCREENING_HYPOTHESIS:
                "Exploratory screening hypothesis",
            TargetRelevance.D_UNSUPPORTED:
                "Unsupported target relationship",
        }[self]

    @property
    def may_publish_affinity(self) -> bool:
        """Whether a numeric affinity may be shown for this class.

        Class C deliberately may not. A resemblance to known ligands justifies
        an experiment, not a reported affinity, and a number attached to an
        unestablished interaction is read as evidence that it occurs.
        """
        return self in (
            TargetRelevance.A_DOCUMENTED_ENGAGEMENT,
            TargetRelevance.B_VALIDATED_DOMAIN_PREDICTION,
        )

    @property
    def may_create_graph_edge(self) -> bool:
        """Only a measurement may become an interaction edge."""
        return self is TargetRelevance.A_DOCUMENTED_ENGAGEMENT


@dataclass
class TargetEvidence:
    """Why a compound was placed in its target-relevance class."""

    relevance: TargetRelevance
    target: str
    has_measurement: bool = False
    measured_pki: float | None = None
    measurement_source: str | None = None

    max_similarity_to_actives: float | None = None
    nearest_active_id: str | None = None
    nearest_active_pki: float | None = None

    total_publications: int | None = None
    target_publications: int | None = None
    target_publication_fraction: float | None = None
    literature_supports_target: bool | None = None

    rationale: str = ""
    caveats: list[str] = field(default_factory=list)

    @property
    def may_publish_affinity(self) -> bool:
        return self.relevance.may_publish_affinity

    @property
    def may_create_graph_edge(self) -> bool:
        return self.relevance.may_create_graph_edge


class TargetRelevanceAssessor:
    """Assigns a target-relevance class from measurement, structure and literature."""

    def __init__(self, pubmed: PubMedClient | None = None,
                 target: dict = MC4R_TARGET) -> None:
        self.pubmed = pubmed or PubMedClient()
        self.target = target
        self._lit_cache: dict[tuple[str, ...], tuple[int, int]] = {}

    # -- literature --------------------------------------------------------
    def literature_association(
        self, names: Sequence[str]
    ) -> tuple[int | None, int | None]:
        """``(total_publications, publications mentioning the target)``.

        Returns ``(None, None)`` when the compound has no name to query, which
        is distinct from measuring zero.
        """
        names = tuple(literature_names([n for n in names if n]))
        if not names:
            return None, None
        if names in self._lit_cache:
            return self._lit_cache[names]
        try:
            _, total, _ = self.pubmed.search(build_query(names), retmax=1)
            _, tgt, _ = self.pubmed.search(
                build_query(names, extra=self.target["literature_terms"]), retmax=1
            )
        except Exception as exc:
            logger.warning("literature lookup failed for %s: %s", names, exc)
            return None, None
        self._lit_cache[names] = (total, tgt)
        return total, tgt

    # -- assessment --------------------------------------------------------
    def assess(
        self,
        *,
        names: Sequence[str] = (),
        measured_pki: float | None = None,
        measurement_source: str | None = None,
        max_similarity_to_actives: float | None = None,
        nearest_active_id: str | None = None,
        nearest_active_pki: float | None = None,
        in_applicability_domain: bool = False,
        check_literature: bool = True,
    ) -> TargetEvidence:
        target_name = f"{self.target['target_chembl_id']} — {self.target['pref_name']}"

        # A: a measurement settles it.
        if measured_pki is not None:
            return TargetEvidence(
                relevance=TargetRelevance.A_DOCUMENTED_ENGAGEMENT,
                target=target_name,
                has_measurement=True,
                measured_pki=measured_pki,
                measurement_source=measurement_source,
                rationale=(
                    "A binding affinity for this molecule at this target has been "
                    "measured and reported. This is direct experimental evidence "
                    "of engagement."
                ),
            )

        total = tgt = None
        supports = None
        if check_literature and names:
            total, tgt = self.literature_association(names)
            if total is not None and tgt is not None:
                fraction = (tgt / total) if total else 0.0
                supports = tgt >= MIN_TARGET_PAPERS and fraction >= MIN_TARGET_FRACTION

        fraction = (tgt / total) if (total and tgt is not None) else None
        common = dict(
            target=target_name,
            max_similarity_to_actives=max_similarity_to_actives,
            nearest_active_id=nearest_active_id,
            nearest_active_pki=nearest_active_pki,
            total_publications=total,
            target_publications=tgt,
            target_publication_fraction=fraction,
            literature_supports_target=supports,
        )

        # D: outside the domain the model was fitted on.
        if not in_applicability_domain:
            return TargetEvidence(
                relevance=TargetRelevance.D_UNSUPPORTED,
                rationale=(
                    "The molecule lies outside the model's applicability domain, "
                    "and no measured activity at this target exists. There is no "
                    "basis for asserting any relationship."
                ),
                **common,
            )

        close = (
            max_similarity_to_actives is not None
            and max_similarity_to_actives >= FAMILY_SIMILARITY_FLOOR
        )

        # B: inside the domain and close to a documented active. Literature can
        # VETO this (an assessed compound whose publications never mention the
        # target is not a validated-domain case), but its *absence* does not
        # block: a bare SMILES has no name to search, and a 0.95-Tanimoto
        # analogue of a documented ligand is itself family-level evidence.
        if close and supports is not False:
            return TargetEvidence(
                relevance=TargetRelevance.B_VALIDATED_DOMAIN_PREDICTION,
                rationale=(
                    f"No measurement for this molecule, but it resembles a "
                    f"documented active ligand ({nearest_active_id}, Tanimoto "
                    f"{max_similarity_to_actives:.2f})"
                    + (
                        f", and {tgt} of {total} publications about it concern "
                        "this target"
                        if supports
                        else ", and no literature was available to check against"
                    )
                    + ". A prediction here is an interpolation within the "
                    "validated domain."
                ),
                caveats=[
                    "A prediction remains an inference, not a measurement, and "
                    "does not establish engagement under physiological conditions.",
                ],
                **common,
            )

        # C: structurally reachable, but target relevance is not established.
        reasons = []
        if close and supports is False:
            reasons.append(
                f"although it resembles a documented active ({max_similarity_to_actives:.2f} "
                f"Tanimoto), only {tgt} of {total} publications about it mention "
                "this target, so the resemblance is not corroborated"
            )
        if not close:
            reasons.append(
                f"its closest documented active ligand is only "
                f"{(max_similarity_to_actives or 0):.2f} Tanimoto away, below the "
                f"{FAMILY_SIMILARITY_FLOOR:.2f} needed to call it the same chemotype"
            )
        if supports is False:
            reasons.append(
                f"only {tgt} of {total} publications about it mention this target "
                "family, which is consistent with incidental co-citation rather "
                "than studied pharmacology"
            )
        elif supports is None:
            reasons.append("its literature could not be assessed")

        return TargetEvidence(
            relevance=TargetRelevance.C_EXPLORATORY_SCREENING_HYPOTHESIS,
            rationale=(
                "The molecule is inside the model's applicability domain, so a "
                "number can be computed, but target relevance is not established: "
                + "; ".join(reasons)
                + ". A structural resemblance is a reason to run an assay, not "
                "evidence of engagement, so no affinity is reported."
            ),
            caveats=[
                "No affinity value is published for this class, because a number "
                "attached to an unestablished interaction reads as evidence that "
                "the interaction occurs.",
                "This does not enter the knowledge graph as an interaction edge.",
            ],
            **common,
        )
