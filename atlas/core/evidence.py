"""Typed schemas for scientific evidence, provenance and abstention.

Three distinctions are enforced by the type system rather than by convention,
because each one is a place where platforms of this kind routinely mislead:

1. **Evidence class** (A-E). Every assertion ATLAS surfaces carries one. A
   mechanistic hypothesis and a measured observation cannot be stored in a way
   that lets the UI render them identically.

2. **Negative result vs absent result.** ``Finding.NO_CHANGE`` means an
   experiment was run and found no effect; ``Finding.NOT_REPORTED`` means
   nobody measured it. Collapsing these is how "no known side effects" gets
   manufactured out of an empty literature.

3. **Abstention as a value.** An analysis that cannot be supported returns an
   :class:`Abstention` carrying a machine-readable reason, not ``None`` and not
   a wide interval around a fabricated point estimate.
"""

from __future__ import annotations

import enum
import hashlib
import json
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from atlas.core.units import Quantity, UnitError, parse_unit, quantity

__all__ = [
    "EvidenceClass",
    "Finding",
    "Species",
    "StudyDesign",
    "AbstentionReason",
    "Abstention",
    "SourceRecord",
    "ExperimentalContext",
    "Observation",
    "EvidenceBundle",
    "utc_now",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceClass(str, enum.Enum):
    """Provenance class of an assertion. Ordered strongest to weakest."""

    DIRECT_EXPERIMENTAL = "A_direct_experimental"
    REPRODUCED_MODEL = "B_reproduced_published_model"
    EVALUATED_INFERENCE = "C_independently_evaluated_inference"
    EXPLORATORY_HYPOTHESIS = "D_exploratory_mechanistic_hypothesis"
    INSUFFICIENT = "E_insufficient_evidence"

    @property
    def rank(self) -> int:
        return {
            "A_direct_experimental": 0,
            "B_reproduced_published_model": 1,
            "C_independently_evaluated_inference": 2,
            "D_exploratory_mechanistic_hypothesis": 3,
            "E_insufficient_evidence": 4,
        }[self.value]

    @property
    def label(self) -> str:
        return {
            EvidenceClass.DIRECT_EXPERIMENTAL: "Direct experimental evidence",
            EvidenceClass.REPRODUCED_MODEL: "Reproduced published model",
            EvidenceClass.EVALUATED_INFERENCE: "Independently evaluated inference",
            EvidenceClass.EXPLORATORY_HYPOTHESIS: "Exploratory mechanistic hypothesis",
            EvidenceClass.INSUFFICIENT: "Insufficient evidence",
        }[self]

    @property
    def is_measured(self) -> bool:
        """True only for classes grounded in an actual measurement."""
        return self is EvidenceClass.DIRECT_EXPERIMENTAL


class Finding(str, enum.Enum):
    """Direction of an observed effect.

    ``NOT_REPORTED`` and ``NOT_MEASURED`` are absences of data.
    ``NO_CHANGE`` is a measured null result and is scientifically informative.
    """

    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"
    MIXED = "mixed_or_context_dependent"
    NOT_REPORTED = "not_reported"
    NOT_MEASURED = "not_measured"

    @property
    def is_absence_of_data(self) -> bool:
        return self in (Finding.NOT_REPORTED, Finding.NOT_MEASURED)


class Species(str, enum.Enum):
    HUMAN = "human"
    MOUSE = "mouse"
    RAT = "rat"
    RABBIT = "rabbit"
    DOG = "dog"
    PIG = "pig"
    MONKEY = "non_human_primate"
    ZEBRAFISH = "zebrafish"
    C_ELEGANS = "c_elegans"
    YEAST = "yeast"
    IN_VITRO = "in_vitro_cell_free_or_cell_line"
    OTHER = "other"
    UNSPECIFIED = "unspecified"

    @property
    def is_human(self) -> bool:
        return self is Species.HUMAN


class StudyDesign(str, enum.Enum):
    RCT = "randomized_controlled_trial"
    CONTROLLED_NONRANDOM = "controlled_non_randomized"
    SINGLE_ARM = "single_arm_interventional"
    OBSERVATIONAL_COHORT = "observational_cohort"
    CASE_CONTROL = "case_control"
    CASE_REPORT = "case_report_or_series"
    IN_VIVO_ANIMAL = "in_vivo_animal_experiment"
    IN_VITRO = "in_vitro_experiment"
    IN_SILICO = "in_silico_model"
    REVIEW_NARRATIVE = "narrative_review"
    REVIEW_SYSTEMATIC = "systematic_review_or_meta_analysis"
    PREPRINT_UNSPECIFIED = "preprint_design_unspecified"
    UNSPECIFIED = "unspecified"

    @property
    def is_primary_experimental(self) -> bool:
        """Reviews restate evidence; they are not themselves measurements."""
        return self not in (
            StudyDesign.REVIEW_NARRATIVE,
            StudyDesign.REVIEW_SYSTEMATIC,
            StudyDesign.IN_SILICO,
            StudyDesign.UNSPECIFIED,
        )


class AbstentionReason(str, enum.Enum):
    """Machine-readable grounds for declining to produce a prediction."""

    UNRESOLVED_MOLECULAR_IDENTITY = "unresolved_molecular_identity"
    NO_SOURCE_EVIDENCE = "no_source_evidence_retrieved"
    INSUFFICIENT_SAMPLE = "insufficient_sample_size"
    NO_HUMAN_DATA = "no_human_data_for_requested_context"
    SPECIES_EXTRAPOLATION_UNSUPPORTED = "species_extrapolation_unsupported"
    ROUTE_NOT_CHARACTERIZED = "administration_route_not_characterized"
    PARAMETERS_UNIDENTIFIABLE = "model_parameters_unidentifiable"
    OUT_OF_APPLICABILITY_DOMAIN = "out_of_model_applicability_domain"
    UNCALIBRATED_UNCERTAINTY = "uncertainty_not_calibrated_for_endpoint"
    CONFLICTING_EVIDENCE = "conflicting_evidence_unresolved"
    FORMULATION_IDENTITY_UNVERIFIED = "formulation_identity_unverified"
    COMBINATION_NOT_STUDIED = "combination_not_directly_studied"
    ENDPOINT_NOT_VALIDATED = "endpoint_model_not_validated"
    CAUSAL_EFFECT_NOT_IDENTIFIABLE = "causal_effect_not_identifiable"
    OUT_OF_SCOPE = "out_of_platform_scope"


class Abstention(BaseModel):
    """A structured refusal to predict. A successful scientific result."""

    model_config = ConfigDict(frozen=True)

    evidence_class: Literal[EvidenceClass.INSUFFICIENT] = EvidenceClass.INSUFFICIENT
    reason: AbstentionReason
    endpoint: str = Field(description="What was asked for, in plain terms.")
    explanation: str = Field(
        min_length=1,
        description="Why the evidence or model cannot support an answer.",
    )
    what_would_resolve_it: list[str] = Field(
        default_factory=list,
        description="Measurements or data that would make the question answerable.",
    )
    examined_sources: list[str] = Field(
        default_factory=list,
        description="Source identifiers consulted before abstaining.",
    )
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def is_abstention(self) -> bool:
        return True


class SourceRecord(BaseModel):
    """A retrieved scientific source with retrieval provenance.

    ``content_sha256`` is computed over the payload actually parsed, so a later
    run can prove whether a source changed underneath a stored conclusion.
    """

    model_config = ConfigDict(frozen=True)

    source_db: str = Field(description="Database the record came from, e.g. 'pubmed'.")
    source_id: str = Field(description="Primary accession within that database.")
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    url: str | None = None
    title: str | None = None
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_date: date | None = None
    publication_types: list[str] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=utc_now)
    retrieval_endpoint: str | None = Field(
        default=None, description="Exact URL or API route used."
    )
    license_note: str | None = Field(
        default=None,
        description="Known licensing/terms constraint on reuse of this record.",
    )
    content_sha256: str | None = None
    is_retracted: bool | None = Field(
        default=None,
        description="None means retraction status was not checked, not that it is clean.",
    )

    @property
    def key(self) -> str:
        return f"{self.source_db}:{self.source_id}"

    @property
    def citation(self) -> str:
        bits: list[str] = []
        if self.authors:
            bits.append(
                self.authors[0] if len(self.authors) == 1 else f"{self.authors[0]} et al."
            )
        if self.publication_date:
            bits.append(str(self.publication_date.year))
        if self.title:
            bits.append(self.title.rstrip("."))
        if self.journal:
            bits.append(self.journal)
        if self.doi:
            bits.append(f"doi:{self.doi}")
        elif self.pmid:
            bits.append(f"PMID:{self.pmid}")
        return ". ".join(bits) if bits else self.key

    @staticmethod
    def hash_payload(payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ExperimentalContext(BaseModel):
    """The biological and experimental conditions a measurement applies to.

    A value without this context is not reusable evidence: 10 nM in a cell-free
    binding assay and 10 nM in human plasma are different facts.
    """

    model_config = ConfigDict(frozen=True)

    species: Species = Species.UNSPECIFIED
    tissue: str | None = None
    cell_type: str | None = None
    assay: str | None = None
    route: str | None = None
    design: StudyDesign = StudyDesign.UNSPECIFIED
    n_subjects: int | None = Field(default=None, ge=0)
    population: str | None = None
    comparator: str | None = None
    duration_value: float | None = None
    duration_unit: str | None = None

    @field_validator("duration_unit")
    @classmethod
    def _check_duration_unit(cls, v: str | None) -> str | None:
        if v is None:
            return None
        _, dim = parse_unit(v)
        if dim.time != 1 or dim.mass or dim.length or dim.substance:
            raise ValueError(f"duration_unit must be a time unit, got {v!r}")
        return v

    @property
    def duration(self) -> Quantity | None:
        if self.duration_value is None or self.duration_unit is None:
            return None
        return quantity(self.duration_value, self.duration_unit)


class Observation(BaseModel):
    """One measured or reported endpoint, bound to its source and context."""

    model_config = ConfigDict(frozen=True)

    subject_compound: str = Field(description="Registry key of the compound studied.")
    endpoint: str = Field(description="What was measured, e.g. 'terminal half-life'.")
    endpoint_ontology_id: str | None = None
    finding: Finding
    value: float | None = None
    unit: str | None = None
    uncertainty_kind: str | None = Field(
        default=None, description="e.g. 'sd', 'sem', '95% CI', 'range', 'cv_percent'."
    )
    uncertainty_low: float | None = None
    uncertainty_high: float | None = None
    context: ExperimentalContext = Field(default_factory=ExperimentalContext)
    source: SourceRecord
    evidence_class: EvidenceClass = EvidenceClass.DIRECT_EXPERIMENTAL
    extraction_method: str = Field(
        default="manual",
        description="How the value left the paper: 'manual', 'api', 'llm_assisted'.",
    )
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    verbatim_quote: str | None = Field(
        default=None, description="Source text the value was taken from, if available."
    )
    limitations: list[str] = Field(default_factory=list)
    recorded_at: datetime = Field(default_factory=utc_now)

    @field_validator("unit")
    @classmethod
    def _unit_parses(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            parse_unit(v)
        except UnitError as exc:
            raise ValueError(f"unparseable unit {v!r}: {exc}") from exc
        return v

    @model_validator(mode="after")
    def _value_consistency(self) -> "Observation":
        has_value = self.value is not None
        if has_value and self.unit is None:
            raise ValueError("a numeric value requires a unit (use '1' if truly unitless)")
        if self.finding.is_absence_of_data and has_value:
            raise ValueError(
                f"finding={self.finding.value} asserts no data exists, but a value "
                "was supplied; use NO_CHANGE for a measured null result"
            )
        if self.evidence_class.is_measured and not has_value:
            if self.finding.is_absence_of_data:
                raise ValueError(
                    "an observation with no data cannot be class A direct "
                    "experimental evidence"
                )
        if (
            self.uncertainty_low is not None
            and self.uncertainty_high is not None
            and self.uncertainty_low > self.uncertainty_high
        ):
            raise ValueError("uncertainty_low exceeds uncertainty_high")
        if (
            self.uncertainty_low is not None or self.uncertainty_high is not None
        ) and self.uncertainty_kind is None:
            raise ValueError("uncertainty bounds require uncertainty_kind")
        if self.extraction_method == "llm_assisted" and not self.verbatim_quote:
            raise ValueError(
                "llm_assisted extraction requires a verbatim_quote so the value "
                "can be checked against the source"
            )
        return self

    @property
    def quantity(self) -> Quantity | None:
        if self.value is None or self.unit is None:
            return None
        return quantity(self.value, self.unit)

    @property
    def is_human_primary(self) -> bool:
        return self.context.species.is_human and self.context.design.is_primary_experimental


class EvidenceBundle(BaseModel):
    """All observations gathered for one question, with a derived class.

    The bundle's ``evidence_class`` is the strongest class present among its
    observations, so a bundle can never be labelled more strongly than its
    best constituent measurement.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    observations: list[Observation] = Field(default_factory=list)
    abstention: Abstention | None = None
    snapshot_id: str | None = Field(
        default=None, description="Evidence ledger snapshot these were read from."
    )

    @model_validator(mode="after")
    def _either_evidence_or_abstention(self) -> "EvidenceBundle":
        if not self.observations and self.abstention is None:
            raise ValueError(
                "an empty bundle must carry an Abstention explaining the absence"
            )
        if self.observations and self.abstention is not None:
            raise ValueError("a bundle cannot both hold observations and abstain")
        return self

    @property
    def evidence_class(self) -> EvidenceClass:
        if self.abstention is not None:
            return EvidenceClass.INSUFFICIENT
        return min((o.evidence_class for o in self.observations), key=lambda c: c.rank)

    @property
    def sources(self) -> list[SourceRecord]:
        seen: dict[str, SourceRecord] = {}
        for obs in self.observations:
            seen.setdefault(obs.source.key, obs.source)
        return list(seen.values())

    @property
    def human_observations(self) -> list[Observation]:
        return [o for o in self.observations if o.is_human_primary]

    def conflicts(self) -> list[tuple[Observation, Observation]]:
        """Pairs of observations on the same endpoint reporting opposite directions.

        Conflict detection is restricted to comparable context (same species and
        assay); a mouse result disagreeing with a human result is not a conflict,
        it is a species difference.
        """
        opposed = {Finding.INCREASE: Finding.DECREASE, Finding.DECREASE: Finding.INCREASE}
        out: list[tuple[Observation, Observation]] = []
        for i, a in enumerate(self.observations):
            for b in self.observations[i + 1 :]:
                if a.endpoint != b.endpoint:
                    continue
                if a.context.species != b.context.species:
                    continue
                if a.context.assay != b.context.assay:
                    continue
                if opposed.get(a.finding) == b.finding:
                    out.append((a, b))
        return out
