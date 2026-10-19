"""Tests for the evidence gate.

These pin the behaviours that live querying exposed as wrong, so they cannot
regress: token-matched registry hits must not count as evidence, literature
that merely mentions pharmacokinetics must not unlock a human PK claim, and an
unidentified substance must not inherit anyone else's data.
"""

from __future__ import annotations

import pytest

from atlas.analysis.feasibility import (
    AnalysisKind,
    FeasibilityEngine,
    LiteratureProfile,
    build_query,
)
from atlas.core.evidence import AbstentionReason
from atlas.registry.resolve import IdentityStatus, MolecularIdentity, Modality, ResolvedField
from atlas.registry.seed import SeedEntry


def lit(
    *,
    total=100,
    human=50,
    human_pk=10,
    rct=5,
    registered=20,
    with_results=5,
) -> LiteratureProfile:
    return LiteratureProfile(
        names_queried=("x",),
        total_publications=total,
        human_publications=human,
        human_pk_publications=human_pk,
        randomized_trials=rct,
        registered_trials=registered,
        trials_with_posted_results=with_results,
        queries={"total": "(\"x\"[tiab])"},
    )


def identity(
    status: IdentityStatus = IdentityStatus.RESOLVED,
    *,
    smiles: str | None = "CC",
    caveats: tuple[str, ...] = (),
) -> MolecularIdentity:
    return MolecularIdentity(
        registry_key="k",
        query_name="x",
        status=status,
        modality=Modality.PEPTIDE,
        smiles=ResolvedField(value=smiles, source_db="chembl", source_id="C1")
        if smiles
        else None,
        caveats=caveats,
    )


def entry(**kw) -> SeedEntry:
    base = dict(key="k", name="x", research_area="other_molecular_research")
    base.update(kw)
    return SeedEntry(**base)


ENGINE = FeasibilityEngine(pubmed=object(), ctgov=object())  # no network used


class TestQueryBuilder:
    def test_names_are_or_combined_over_title_and_abstract(self):
        q = build_query(["BPC-157", "BPC 157"])
        assert '"BPC-157"[tiab]' in q and '"BPC 157"[tiab]' in q and " OR " in q

    def test_extra_filter_is_conjoined(self):
        q = build_query(["x"], extra="humans[mh]")
        assert q.startswith("(") and "AND (humans[mh])" in q

    def test_alias_count_is_capped(self):
        q = build_query([f"n{i}" for i in range(20)], max_aliases=3)
        assert q.count("[tiab]") == 4

    def test_brackets_are_stripped_so_the_parser_is_not_broken(self):
        assert "[" not in build_query(["weird[name]"]).replace("[tiab]", "")

    def test_duplicate_names_collapse(self):
        assert build_query(["x", "x", "x"]).count("[tiab]") == 1


class TestLiteratureProfile:
    def test_registered_trials_are_not_evidence(self):
        """Activity is not evidence: results must be posted."""
        p = lit(registered=4, with_results=0)
        assert not p.has_registry_evidence

    def test_posted_results_are_evidence(self):
        assert lit(registered=4, with_results=1).has_registry_evidence

    def test_density_buckets(self):
        assert lit(total=0).evidence_density == "none"
        assert lit(total=4).evidence_density == "very_sparse"
        assert lit(total=30).evidence_density == "sparse"
        assert lit(total=225).evidence_density == "moderate"
        assert lit(total=5707).evidence_density == "substantial"

    def test_failed_measurement_is_negative_not_zero(self):
        """-1 marks 'not measured'; it must not read as an absence of evidence."""
        p = lit(human_pk=-1)
        assert not p.has_human_pk_evidence
        assert p.human_pk_publications == -1


class TestExploratorySimulationAlwaysAvailable:
    def test_permitted_even_with_no_evidence_at_all(self):
        a = ENGINE.assess(entry(), identity(IdentityStatus.UNRESOLVED, smiles=None),
                          lit(total=0, human=0, human_pk=0, registered=0, with_results=0))
        assert a.is_permitted(AnalysisKind.EXPLORATORY_SIMULATION)

    def test_permitted_for_a_mixture(self):
        a = ENGINE.assess(
            entry(is_mixture=True, declared_modality=Modality.MIXTURE),
            identity(IdentityStatus.MIXTURE, smiles=None),
            lit(total=0),
        )
        assert a.is_permitted(AnalysisKind.EXPLORATORY_SIMULATION)


class TestHumanPKGate:
    def test_abundant_literature_alone_does_not_permit_a_human_claim(self):
        """Semaglutide has 174 human PK records; parameters still must be extracted."""
        a = ENGINE.assess(entry(), identity(), lit(human_pk=174, with_results=110))
        assert not a.is_permitted(AnalysisKind.HUMAN_PK_SIMULATION)
        ab = a.abstention_for(AnalysisKind.HUMAN_PK_SIMULATION)
        assert ab.reason is AbstentionReason.PARAMETERS_UNIDENTIFIABLE
        assert "174" in ab.explanation

    def test_sparse_pk_mentions_do_not_permit_a_human_claim(self):
        """BPC-157 matched three PK-shaped records and no posted results."""
        a = ENGINE.assess(entry(), identity(), lit(human_pk=3, with_results=0))
        ab = a.abstention_for(AnalysisKind.HUMAN_PK_SIMULATION)
        assert ab.reason is AbstentionReason.PARAMETERS_UNIDENTIFIABLE

    def test_extracted_parameters_permit_the_claim(self):
        a = ENGINE.assess(
            entry(), identity(), lit(human_pk=174),
            pk_parameters={"CL": 0.05, "V": 12.5},
        )
        assert a.is_permitted(AnalysisKind.HUMAN_PK_SIMULATION)

    def test_no_human_data_at_all_gives_the_stronger_reason(self):
        a = ENGINE.assess(entry(), identity(), lit(human_pk=0, registered=0, with_results=0))
        assert (
            a.abstention_for(AnalysisKind.HUMAN_PK_SIMULATION).reason
            is AbstentionReason.NO_HUMAN_DATA
        )

    def test_unidentified_substance_cannot_claim_human_pk_even_with_parameters(self):
        """Amino H2O token-matched 23 registry studies; identity still fails."""
        a = ENGINE.assess(
            entry(supplier_defined=True),
            identity(IdentityStatus.UNRESOLVED, smiles=None),
            lit(human_pk=5, registered=23, with_results=7),
            pk_parameters={"CL": 1.0, "V": 10.0},
        )
        assert not a.is_permitted(AnalysisKind.HUMAN_PK_SIMULATION)
        assert (
            a.abstention_for(AnalysisKind.HUMAN_PK_SIMULATION).reason
            is AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY
        )

    def test_supplier_defined_identity_also_blocks_the_claim(self):
        a = ENGINE.assess(
            entry(supplier_defined=True),
            identity(IdentityStatus.SUPPLIER_DEFINED),
            lit(),
            pk_parameters={"CL": 1.0},
        )
        assert not a.is_permitted(AnalysisKind.HUMAN_PK_SIMULATION)


class TestStructureGate:
    def test_resolved_structure_is_permitted(self):
        a = ENGINE.assess(entry(), identity(), lit())
        assert a.is_permitted(AnalysisKind.STRUCTURE_BASED_PREDICTION)

    def test_missing_structure_is_refused(self):
        a = ENGINE.assess(entry(), identity(smiles=None), lit())
        assert not a.is_permitted(AnalysisKind.STRUCTURE_BASED_PREDICTION)
        ab = a.abstention_for(AnalysisKind.STRUCTURE_BASED_PREDICTION)
        assert ab.reason is AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY
        assert ab.what_would_resolve_it

    def test_mixture_is_refused_as_a_formulation_problem(self):
        a = ENGINE.assess(
            entry(is_mixture=True), identity(IdentityStatus.MIXTURE, smiles=None), lit()
        )
        ab = a.abstention_for(AnalysisKind.STRUCTURE_BASED_PREDICTION)
        assert ab.reason is AbstentionReason.FORMULATION_IDENTITY_UNVERIFIED

    def test_ambiguous_identity_refusal_quotes_the_caveat(self):
        a = ENGINE.assess(
            entry(),
            identity(IdentityStatus.AMBIGUOUS, caveats=("weights differ wildly",)),
            lit(),
        )
        ab = a.abstention_for(AnalysisKind.STRUCTURE_BASED_PREDICTION)
        assert "weights differ wildly" in ab.explanation


class TestSafetyIsNeverQuantified:
    def test_always_refused_regardless_of_evidence(self):
        a = ENGINE.assess(
            entry(), identity(), lit(total=50000, human_pk=9999, with_results=9999)
        )
        assert not a.is_permitted(AnalysisKind.QUANTITATIVE_SAFETY_PREDICTION)

    def test_refusal_explains_the_denominator_problem(self):
        a = ENGINE.assess(entry(), identity(), lit())
        ab = a.abstention_for(AnalysisKind.QUANTITATIVE_SAFETY_PREDICTION)
        assert "denominator" in ab.explanation
        assert ab.reason is AbstentionReason.ENDPOINT_NOT_VALIDATED


class TestCombinationGate:
    def test_mixture_combination_effects_are_refused(self):
        a = ENGINE.assess(
            entry(is_mixture=True, reported_components=("ghk-cu", "bpc-157")),
            identity(IdentityStatus.MIXTURE, smiles=None),
            lit(),
        )
        ab = a.abstention_for(AnalysisKind.COMBINATION_EFFECT_PREDICTION)
        assert ab.reason is AbstentionReason.COMBINATION_NOT_STUDIED
        assert "ghk-cu" in ab.explanation

    def test_refusal_states_components_do_not_add(self):
        a = ENGINE.assess(
            entry(is_mixture=True, reported_components=("a-key",)),
            identity(IdentityStatus.MIXTURE, smiles=None),
            lit(),
        )
        ab = a.abstention_for(AnalysisKind.COMBINATION_EFFECT_PREDICTION)
        assert "cannot be added" in ab.explanation

    def test_single_entity_is_not_asked_about_combinations(self):
        a = ENGINE.assess(entry(), identity(), lit())
        assert AnalysisKind.COMBINATION_EFFECT_PREDICTION not in a.refusals


class TestLiteratureGate:
    def test_zero_literature_refuses_summary_and_hypothesis(self):
        a = ENGINE.assess(entry(), identity(), lit(total=0))
        assert not a.is_permitted(AnalysisKind.LITERATURE_SUMMARY)
        assert not a.is_permitted(AnalysisKind.MECHANISM_HYPOTHESIS)

    def test_any_literature_permits_both(self):
        a = ENGINE.assess(entry(), identity(), lit(total=1))
        assert a.is_permitted(AnalysisKind.LITERATURE_SUMMARY)
        assert a.is_permitted(AnalysisKind.MECHANISM_HYPOTHESIS)


class TestAssessmentReporting:
    def test_every_refusal_carries_sources_and_resolvers(self):
        a = ENGINE.assess(
            entry(is_mixture=True, reported_components=("c",)),
            identity(IdentityStatus.MIXTURE, smiles=None),
            lit(total=0),
        )
        assert a.refusals
        for ab in a.refusals.values():
            assert ab.explanation
            assert ab.what_would_resolve_it, ab.reason
            assert ab.examined_sources

    def test_permitted_and_refused_do_not_overlap(self):
        a = ENGINE.assess(entry(), identity(), lit())
        assert not (set(a.permitted) & set(a.refusals))

    def test_summary_line_reports_real_counts(self):
        a = ENGINE.assess(entry(), identity(), lit(total=225, human_pk=3, with_results=0))
        line = a.summary_line()
        assert "225 pubs" in line and "3 human PK" in line
