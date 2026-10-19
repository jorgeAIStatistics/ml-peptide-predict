"""Tests for the evidence schemas.

The point of these is to confirm that the schema refuses to represent the
specific misleading states this platform must never produce.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from atlas.core.evidence import (
    Abstention,
    AbstentionReason,
    EvidenceBundle,
    EvidenceClass,
    ExperimentalContext,
    Finding,
    Observation,
    SourceRecord,
    Species,
    StudyDesign,
)


def src(**kw) -> SourceRecord:
    base = dict(source_db="pubmed", source_id="12345678", pmid="12345678")
    base.update(kw)
    return SourceRecord(**base)


def obs(**kw) -> Observation:
    base = dict(
        subject_compound="test-compound",
        endpoint="terminal half-life",
        finding=Finding.INCREASE,
        value=7.0,
        unit="h",
        source=src(),
    )
    base.update(kw)
    return Observation(**base)


class TestEvidenceClass:
    def test_ordering_is_strongest_first(self):
        assert EvidenceClass.DIRECT_EXPERIMENTAL.rank < EvidenceClass.REPRODUCED_MODEL.rank
        assert EvidenceClass.EXPLORATORY_HYPOTHESIS.rank < EvidenceClass.INSUFFICIENT.rank

    def test_only_class_a_counts_as_measured(self):
        assert EvidenceClass.DIRECT_EXPERIMENTAL.is_measured
        for other in (
            EvidenceClass.REPRODUCED_MODEL,
            EvidenceClass.EVALUATED_INFERENCE,
            EvidenceClass.EXPLORATORY_HYPOTHESIS,
            EvidenceClass.INSUFFICIENT,
        ):
            assert not other.is_measured

    def test_every_class_has_a_human_label(self):
        for cls in EvidenceClass:
            assert cls.label and not cls.label.startswith(("A_", "B_", "C_"))


class TestNegativeVersusMissing:
    def test_no_change_is_not_absence_of_data(self):
        assert not Finding.NO_CHANGE.is_absence_of_data

    def test_not_reported_is_absence_of_data(self):
        assert Finding.NOT_REPORTED.is_absence_of_data
        assert Finding.NOT_MEASURED.is_absence_of_data

    def test_cannot_attach_a_value_to_a_not_reported_finding(self):
        with pytest.raises(ValidationError, match="asserts no data exists"):
            obs(finding=Finding.NOT_REPORTED, value=7.0, unit="h")

    def test_not_reported_without_value_is_allowed_as_weaker_class(self):
        o = obs(
            finding=Finding.NOT_REPORTED,
            value=None,
            unit=None,
            evidence_class=EvidenceClass.INSUFFICIENT,
        )
        assert o.quantity is None

    def test_absent_data_cannot_be_class_a(self):
        with pytest.raises(ValidationError, match="cannot be class A"):
            obs(
                finding=Finding.NOT_REPORTED,
                value=None,
                unit=None,
                evidence_class=EvidenceClass.DIRECT_EXPERIMENTAL,
            )

    def test_measured_null_result_is_valid_class_a(self):
        o = obs(finding=Finding.NO_CHANGE, value=0.2, unit="1")
        assert o.evidence_class is EvidenceClass.DIRECT_EXPERIMENTAL


class TestObservationValidation:
    def test_value_requires_unit(self):
        with pytest.raises(ValidationError, match="requires a unit"):
            obs(value=7.0, unit=None)

    def test_unparseable_unit_rejected(self):
        with pytest.raises(ValidationError, match="unparseable unit"):
            obs(unit="furlongs per fortnight")

    def test_inverted_uncertainty_rejected(self):
        with pytest.raises(ValidationError, match="exceeds"):
            obs(uncertainty_kind="95% CI", uncertainty_low=9.0, uncertainty_high=2.0)

    def test_uncertainty_bounds_require_a_kind(self):
        with pytest.raises(ValidationError, match="uncertainty_kind"):
            obs(uncertainty_low=1.0, uncertainty_high=2.0)

    def test_llm_extraction_requires_verbatim_quote(self):
        with pytest.raises(ValidationError, match="verbatim_quote"):
            obs(extraction_method="llm_assisted")

    def test_llm_extraction_with_quote_is_accepted(self):
        o = obs(extraction_method="llm_assisted", verbatim_quote="t1/2 was 7.0 h")
        assert o.verbatim_quote

    def test_quantity_round_trips_through_units(self):
        o = obs(value=168.0, unit="h")
        assert o.quantity is not None
        assert o.quantity.to("d").value == pytest.approx(7.0)

    def test_duration_unit_must_be_time(self):
        with pytest.raises(ValidationError, match="time unit"):
            ExperimentalContext(duration_value=4.0, duration_unit="mg")

    def test_negative_sample_size_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentalContext(n_subjects=-3)


class TestHumanPrimaryClassification:
    def test_human_rct_is_human_primary(self):
        o = obs(context=ExperimentalContext(species=Species.HUMAN, design=StudyDesign.RCT))
        assert o.is_human_primary

    def test_human_narrative_review_is_not_primary(self):
        o = obs(
            context=ExperimentalContext(
                species=Species.HUMAN, design=StudyDesign.REVIEW_NARRATIVE
            )
        )
        assert not o.is_human_primary

    def test_mouse_experiment_is_not_human_primary(self):
        o = obs(
            context=ExperimentalContext(
                species=Species.MOUSE, design=StudyDesign.IN_VIVO_ANIMAL
            )
        )
        assert not o.is_human_primary

    def test_in_silico_is_not_primary_experimental(self):
        assert not StudyDesign.IN_SILICO.is_primary_experimental


class TestSourceRecord:
    def test_citation_includes_identifier(self):
        s = src(
            authors=["Smith J", "Doe A"],
            title="A study of something.",
            journal="J Test",
            publication_date=date(2021, 5, 1),
            doi="10.1000/xyz",
        )
        c = s.citation
        assert "Smith J et al." in c and "2021" in c and "doi:10.1000/xyz" in c

    def test_citation_falls_back_to_key(self):
        assert src(pmid=None).citation == "pubmed:12345678"

    def test_retraction_unchecked_is_none_not_false(self):
        assert src().is_retracted is None

    def test_payload_hash_is_order_independent(self):
        a = SourceRecord.hash_payload({"x": 1, "y": [1, 2]})
        b = SourceRecord.hash_payload({"y": [1, 2], "x": 1})
        assert a == b

    def test_payload_hash_changes_with_content(self):
        a = SourceRecord.hash_payload({"x": 1})
        b = SourceRecord.hash_payload({"x": 2})
        assert a != b


class TestAbstention:
    def test_abstention_is_always_class_e(self):
        a = Abstention(
            reason=AbstentionReason.NO_HUMAN_DATA,
            endpoint="human plasma half-life",
            explanation="No human PK study was retrieved for this compound.",
        )
        assert a.evidence_class is EvidenceClass.INSUFFICIENT
        assert a.is_abstention

    def test_abstention_requires_an_explanation(self):
        with pytest.raises(ValidationError):
            Abstention(
                reason=AbstentionReason.NO_SOURCE_EVIDENCE,
                endpoint="x",
                explanation="",
            )


class TestEvidenceBundle:
    def test_empty_bundle_must_abstain(self):
        with pytest.raises(ValidationError, match="must carry an Abstention"):
            EvidenceBundle(question="what is the half-life?")

    def test_bundle_cannot_both_hold_and_abstain(self):
        with pytest.raises(ValidationError, match="cannot both"):
            EvidenceBundle(
                question="q",
                observations=[obs()],
                abstention=Abstention(
                    reason=AbstentionReason.NO_SOURCE_EVIDENCE,
                    endpoint="q",
                    explanation="none",
                ),
            )

    def test_bundle_class_is_the_strongest_present(self):
        b = EvidenceBundle(
            question="q",
            observations=[
                obs(evidence_class=EvidenceClass.EXPLORATORY_HYPOTHESIS),
                obs(evidence_class=EvidenceClass.DIRECT_EXPERIMENTAL),
            ],
        )
        assert b.evidence_class is EvidenceClass.DIRECT_EXPERIMENTAL

    def test_bundle_of_only_hypotheses_is_not_promoted(self):
        b = EvidenceBundle(
            question="q",
            observations=[obs(evidence_class=EvidenceClass.EXPLORATORY_HYPOTHESIS)] * 5,
        )
        assert b.evidence_class is EvidenceClass.EXPLORATORY_HYPOTHESIS

    def test_abstaining_bundle_is_class_e(self):
        b = EvidenceBundle(
            question="q",
            abstention=Abstention(
                reason=AbstentionReason.COMBINATION_NOT_STUDIED,
                endpoint="q",
                explanation="No direct study of the combination was found.",
            ),
        )
        assert b.evidence_class is EvidenceClass.INSUFFICIENT

    def test_sources_are_deduplicated(self):
        b = EvidenceBundle(question="q", observations=[obs(), obs()])
        assert len(b.sources) == 1

    def test_conflict_detected_within_same_context(self):
        ctx = ExperimentalContext(species=Species.MOUSE, assay="qPCR")
        b = EvidenceBundle(
            question="q",
            observations=[
                obs(finding=Finding.INCREASE, context=ctx),
                obs(
                    finding=Finding.DECREASE,
                    context=ctx,
                    source=src(source_id="999", pmid="999"),
                ),
            ],
        )
        assert len(b.conflicts()) == 1

    def test_species_difference_is_not_a_conflict(self):
        b = EvidenceBundle(
            question="q",
            observations=[
                obs(
                    finding=Finding.INCREASE,
                    context=ExperimentalContext(species=Species.MOUSE, assay="qPCR"),
                ),
                obs(
                    finding=Finding.DECREASE,
                    context=ExperimentalContext(species=Species.HUMAN, assay="qPCR"),
                    source=src(source_id="999", pmid="999"),
                ),
            ],
        )
        assert b.conflicts() == []

    def test_human_observations_filter(self):
        b = EvidenceBundle(
            question="q",
            observations=[
                obs(
                    context=ExperimentalContext(
                        species=Species.HUMAN, design=StudyDesign.RCT
                    )
                ),
                obs(
                    context=ExperimentalContext(
                        species=Species.MOUSE, design=StudyDesign.IN_VIVO_ANIMAL
                    ),
                    source=src(source_id="999", pmid="999"),
                ),
            ],
        )
        assert len(b.human_observations) == 1
