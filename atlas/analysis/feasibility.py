"""Evidence gating: decide what may be computed for a compound, and refuse the rest.

This is the engine that makes a refusal a result. For a given registry entry it
measures how much evidence actually exists -- by querying PubMed and
ClinicalTrials.gov, not by consulting any stored opinion -- and then decides,
per analysis type, whether the platform may proceed.

Two counts are kept strictly apart. Registered trials indicate research
activity; trials with *posted results* are evidence. A compound with four
registered studies and zero posted results has generated no registry evidence,
and the gate says so.

Every refusal is an :class:`Abstention` carrying the reason, the queries that
were run, and what would resolve it. Nothing is estimated to fill a gap.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from atlas.core.evidence import Abstention, AbstentionReason
from atlas.registry.resolve import IdentityStatus, MolecularIdentity
from atlas.registry.seed import SeedEntry
from atlas.sources.ctgov import ClinicalTrialsClient
from atlas.sources.pubmed import PubMedClient

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisKind",
    "LiteratureProfile",
    "FeasibilityAssessment",
    "FeasibilityEngine",
    "build_query",
]

#: Aliases beyond this count are dropped from queries; long OR-chains slow
#: E-utilities and add little recall.
MAX_ALIASES_IN_QUERY = 4


class AnalysisKind(str, enum.Enum):
    """Analyses the platform can be asked to perform, in increasing demand."""

    LITERATURE_SUMMARY = "literature_summary"
    MECHANISM_HYPOTHESIS = "mechanism_hypothesis"
    #: A mathematical run on user-supplied parameters, labelled as exploratory.
    #: Always available, because it makes no claim about any organism.
    EXPLORATORY_SIMULATION = "exploratory_simulation"
    STRUCTURE_BASED_PREDICTION = "structure_based_prediction"
    #: A simulation presented as describing human pharmacokinetics. Requires
    #: source-backed parameters, not merely literature that mentions PK.
    HUMAN_PK_SIMULATION = "human_pk_simulation"
    QUANTITATIVE_SAFETY_PREDICTION = "quantitative_safety_prediction"
    COMBINATION_EFFECT_PREDICTION = "combination_effect_prediction"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


def _quote(term: str) -> str:
    """Quote a term for PubMed, stripping characters that break its parser."""
    cleaned = term.replace('"', "").replace("[", "").replace("]", "").strip()
    return f'"{cleaned}"'


def build_query(
    names: Sequence[str], *, extra: str | None = None, max_aliases: int = MAX_ALIASES_IN_QUERY
) -> str:
    """Build a PubMed query matching any of ``names`` in title or abstract."""
    chosen = list(dict.fromkeys(n for n in names if n))[: max_aliases + 1]
    clause = " OR ".join(f"{_quote(n)}[tiab]" for n in chosen)
    query = f"({clause})"
    if extra:
        query = f"{query} AND ({extra})"
    return query


# PubMed filters. ``humans[mh]`` is the MeSH human-subject tag and is far more
# precise than a free-text search for "human".
_HUMAN_FILTER = "humans[mh]"
_PK_FILTER = (
    "pharmacokinetics[tiab] OR pharmacokinetic[tiab] OR "
    '"half-life"[tiab] OR clearance[tiab] OR "area under the curve"[tiab] OR '
    "pharmacokinetics[mh]"
)
_RCT_FILTER = "randomized controlled trial[pt]"


@dataclass(frozen=True)
class LiteratureProfile:
    """Measured evidence volume for one compound. All counts are real hits."""

    names_queried: tuple[str, ...]
    total_publications: int
    human_publications: int
    human_pk_publications: int
    randomized_trials: int
    registered_trials: int
    trials_with_posted_results: int
    queries: dict[str, str] = field(default_factory=dict)

    @property
    def has_any_literature(self) -> bool:
        return self.total_publications > 0

    @property
    def has_human_evidence(self) -> bool:
        return self.human_publications > 0

    @property
    def has_human_pk_evidence(self) -> bool:
        return self.human_pk_publications > 0

    @property
    def has_registry_evidence(self) -> bool:
        """Registered trials are activity; posted results are evidence."""
        return self.trials_with_posted_results > 0

    @property
    def evidence_density(self) -> str:
        """A coarse, honest descriptor of literature volume."""
        n = self.total_publications
        if n == 0:
            return "none"
        if n < 10:
            return "very_sparse"
        if n < 50:
            return "sparse"
        if n < 250:
            return "moderate"
        return "substantial"


@dataclass(frozen=True)
class FeasibilityAssessment:
    """What may and may not be computed for one compound, with reasons."""

    registry_key: str
    display_name: str
    identity_status: IdentityStatus
    literature: LiteratureProfile
    permitted: tuple[AnalysisKind, ...]
    refusals: dict[AnalysisKind, Abstention]

    def is_permitted(self, kind: AnalysisKind) -> bool:
        return kind in self.permitted

    def abstention_for(self, kind: AnalysisKind) -> Abstention | None:
        return self.refusals.get(kind)

    @property
    def n_refused(self) -> int:
        return len(self.refusals)

    def summary_line(self) -> str:
        return (
            f"{self.display_name}: identity={self.identity_status.value}, "
            f"literature={self.literature.evidence_density} "
            f"({self.literature.total_publications} pubs, "
            f"{self.literature.human_pk_publications} human PK, "
            f"{self.literature.trials_with_posted_results} trials with results), "
            f"{len(self.permitted)} permitted / {self.n_refused} refused"
        )


class FeasibilityEngine:
    """Measures available evidence and gates analyses on it."""

    def __init__(
        self,
        pubmed: PubMedClient | None = None,
        ctgov: ClinicalTrialsClient | None = None,
    ) -> None:
        self.pubmed = pubmed or PubMedClient()
        self.ctgov = ctgov or ClinicalTrialsClient()

    # -- measurement --------------------------------------------------------
    def profile_literature(
        self, name: str, aliases: Iterable[str] = ()
    ) -> LiteratureProfile:
        """Count real hits across PubMed and ClinicalTrials.gov."""
        names = (name, *aliases)
        queries = {
            "total": build_query(names),
            "human": build_query(names, extra=_HUMAN_FILTER),
            "human_pk": build_query(names, extra=f"({_HUMAN_FILTER}) AND ({_PK_FILTER})"),
            "rct": build_query(names, extra=_RCT_FILTER),
        }

        counts: dict[str, int] = {}
        for label, query in queries.items():
            try:
                _, total, _ = self.pubmed.search(query, retmax=1)
                counts[label] = total
            except Exception as exc:
                logger.warning("PubMed count failed (%s) for %r: %s", label, name, exc)
                counts[label] = -1  # -1 marks "not measured", never "zero"

        try:
            registered = self.ctgov.count(name)
            with_results = self.ctgov.count_with_results(name)
        except Exception as exc:
            logger.warning("ClinicalTrials.gov count failed for %r: %s", name, exc)
            registered = with_results = -1

        return LiteratureProfile(
            names_queried=tuple(names),
            total_publications=counts["total"],
            human_publications=counts["human"],
            human_pk_publications=counts["human_pk"],
            randomized_trials=counts["rct"],
            registered_trials=registered,
            trials_with_posted_results=with_results,
            queries=queries,
        )

    # -- gating -------------------------------------------------------------
    def assess(
        self,
        entry: SeedEntry,
        identity: MolecularIdentity,
        literature: LiteratureProfile | None = None,
        *,
        pk_parameters: Mapping[str, object] | None = None,
    ) -> FeasibilityAssessment:
        """Gate each analysis kind for one compound.

        ``pk_parameters`` holds source-backed pharmacokinetic parameters already
        extracted and verified for this compound and route. It is deliberately
        separate from the literature counts: the counts say whether such
        parameters are plausibly findable, while only the extracted parameters
        permit a simulation to be presented as describing humans.
        """
        lit = literature or self.profile_literature(entry.name, entry.aliases)
        permitted: list[AnalysisKind] = []
        refusals: dict[AnalysisKind, Abstention] = {}
        sources = [f"pubmed query: {q}" for q in lit.queries.values()]
        sources.append(f"clinicaltrials.gov term: {entry.name}")

        def refuse(
            kind: AnalysisKind,
            reason: AbstentionReason,
            explanation: str,
            resolvers: list[str],
        ) -> None:
            refusals[kind] = Abstention(
                reason=reason,
                endpoint=f"{kind.label} for {entry.name}",
                explanation=explanation,
                what_would_resolve_it=resolvers,
                examined_sources=sources,
            )

        # Literature summary: permitted whenever any literature exists. There is
        # nothing to summarise otherwise.
        if lit.has_any_literature:
            permitted.append(AnalysisKind.LITERATURE_SUMMARY)
        else:
            refuse(
                AnalysisKind.LITERATURE_SUMMARY,
                AbstentionReason.NO_SOURCE_EVIDENCE,
                "No PubMed records matched this compound's name or aliases, so "
                "there is no literature to summarise.",
                [
                    "Confirm the compound's correct chemical name or a database "
                    "identifier, then re-run retrieval.",
                    "Search supplier documentation or patents, which are outside "
                    "the sources queried here.",
                ],
            )

        # Mechanistic hypothesis: needs literature, and is explicitly labelled a
        # hypothesis rather than a finding.
        if lit.has_any_literature:
            permitted.append(AnalysisKind.MECHANISM_HYPOTHESIS)
        else:
            refuse(
                AnalysisKind.MECHANISM_HYPOTHESIS,
                AbstentionReason.NO_SOURCE_EVIDENCE,
                "A mechanistic hypothesis must be anchored to at least one "
                "retrievable publication.",
                ["Establish the molecular identity so the literature can be found."],
            )

        # Structure-based prediction: requires a confidently resolved structure.
        if not identity.blocks_structure_models():
            permitted.append(AnalysisKind.STRUCTURE_BASED_PREDICTION)
        else:
            if identity.status is IdentityStatus.MIXTURE:
                reason = AbstentionReason.FORMULATION_IDENTITY_UNVERIFIED
                explanation = (
                    "This entry is a formulation rather than a single molecular "
                    "entity. A structure-based model has no single structure to "
                    "act on, and component ratios are not standardised."
                )
                resolvers = [
                    "Obtain a per-product certificate of analysis giving each "
                    "component's identity and amount.",
                    "Model each identified constituent separately, without "
                    "combining the results into a mixture-level claim.",
                ]
            elif identity.status is IdentityStatus.AMBIGUOUS:
                reason = AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY
                explanation = (
                    "Databases hold structurally different molecules under this "
                    "name, so it is not determined which entity a prediction "
                    "would describe. "
                    + " ".join(identity.caveats)
                )
                resolvers = [
                    "Specify which molecular form is intended, by database "
                    "identifier or exact sequence.",
                ]
            elif identity.status is IdentityStatus.SUPPLIER_DEFINED:
                reason = AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY
                explanation = (
                    "The identity of this material is defined by its supplier "
                    "rather than by a reference database, so no verified "
                    "structure is available."
                )
                resolvers = [
                    "Obtain the sequence or structure from a certificate of "
                    "analysis or analytical characterisation.",
                ]
            else:
                reason = AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY
                explanation = (
                    "No structure was resolved from ChEMBL or PubChem for this "
                    "name or its aliases, so there is nothing to featurise."
                )
                resolvers = [
                    "Supply a SMILES string, InChI or peptide sequence.",
                    "Provide a ChEMBL or PubChem identifier for the intended entity.",
                ]
            refuse(AnalysisKind.STRUCTURE_BASED_PREDICTION, reason, explanation, resolvers)

        # An exploratory run makes no claim about any organism, so it is always
        # available. What it may be *called* is the constrained part.
        permitted.append(AnalysisKind.EXPLORATORY_SIMULATION)

        # Human PK simulation: a solver will integrate any parameters handed to
        # it. Presenting the result as human pharmacokinetics requires
        # source-backed parameters for this compound and route. Counting
        # publications that merely mention pharmacokinetics is not enough: a
        # review discussing humans cannot parameterise a model, and BPC-157
        # matches three such records while no extracted parameters exist.
        identity_usable = identity.status in (
            IdentityStatus.RESOLVED,
            IdentityStatus.AMBIGUOUS,
        )
        if pk_parameters and identity_usable:
            permitted.append(AnalysisKind.HUMAN_PK_SIMULATION)
        elif not identity_usable:
            refuse(
                AnalysisKind.HUMAN_PK_SIMULATION,
                AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY,
                "A simulation cannot be attributed to a substance whose identity "
                f"is {identity.status.value}. Even where registry or literature "
                "matches exist, it is undetermined what they refer to.",
                [
                    "Establish the molecular identity, then extract "
                    "pharmacokinetic parameters for the intended route.",
                ],
            )
        elif not (lit.has_human_pk_evidence or lit.has_registry_evidence):
            refuse(
                AnalysisKind.HUMAN_PK_SIMULATION,
                AbstentionReason.NO_HUMAN_DATA,
                "No human pharmacokinetic literature and no trials with posted "
                "results were found, so clearance and volume of distribution "
                "cannot be grounded in observation. PubMed returned "
                f"{max(lit.human_pk_publications, 0)} human pharmacokinetic "
                f"records; ClinicalTrials.gov holds "
                f"{max(lit.registered_trials, 0)} registered studies of which "
                f"{max(lit.trials_with_posted_results, 0)} have posted results. "
                "A simulation run on assumed parameters would describe the "
                "assumptions, not the compound.",
                [
                    "A human pharmacokinetic study reporting clearance, volume "
                    "of distribution and bioavailability for the intended route.",
                    "An exploratory simulation remains available with "
                    "user-supplied parameters, labelled as exploratory.",
                ],
            )
        else:
            refuse(
                AnalysisKind.HUMAN_PK_SIMULATION,
                AbstentionReason.PARAMETERS_UNIDENTIFIABLE,
                "Candidate human pharmacokinetic literature exists "
                f"({max(lit.human_pk_publications, 0)} PubMed records, "
                f"{max(lit.trials_with_posted_results, 0)} trials with posted "
                "results), but no pharmacokinetic parameters have been extracted "
                "from it and verified against the source. Until clearance, "
                "volume of distribution and bioavailability are recorded as "
                "sourced measurements, the model has nothing compound-specific "
                "to integrate.",
                [
                    "Extract clearance, volume of distribution and bioavailability "
                    "from the identified records, with units and uncertainty, and "
                    "verify each against the source text.",
                    "An exploratory simulation remains available with "
                    "user-supplied parameters, labelled as exploratory.",
                ],
            )

        # Quantitative safety prediction: deliberately never permitted. Spontaneous
        # report counts and sparse trials cannot yield incidence or risk.
        refuse(
            AnalysisKind.QUANTITATIVE_SAFETY_PREDICTION,
            AbstentionReason.ENDPOINT_NOT_VALIDATED,
            "This platform does not produce numerical safety or risk estimates. "
            "Spontaneous adverse-event reports have unknown denominators and "
            "reporting bias, so they cannot be converted into incidence or into "
            "a probability of harm for any individual, and no validated outcome "
            "model exists for these endpoints.",
            [
                "Controlled studies with systematic adverse-event ascertainment "
                "and a defined denominator population.",
                "An externally validated outcome model with a stated context of use.",
            ],
        )

        # Combination effects: require direct study of the combination itself.
        if entry.is_mixture or entry.reported_components:
            components = ", ".join(entry.reported_components) or "its constituents"
            refuse(
                AnalysisKind.COMBINATION_EFFECT_PREDICTION,
                AbstentionReason.COMBINATION_NOT_STUDIED,
                "Evidence for individual components does not establish the "
                f"behaviour of the combination. This entry is reported to contain "
                f"{components}, but component-level findings cannot be added "
                "together: interaction, formulation chemistry and altered "
                "exposure are not predictable from the parts.",
                [
                    "A study administering this specific combination at known "
                    "ratios and measuring the endpoint of interest.",
                    "Formulation-compatibility data for the components in a "
                    "shared vehicle.",
                ],
            )

        return FeasibilityAssessment(
            registry_key=entry.key,
            display_name=entry.name,
            identity_status=identity.status,
            literature=lit,
            permitted=tuple(permitted),
            refusals=refusals,
        )
