"""ClinicalTrials.gov API v2 access.

The registry is used for two distinct purposes, and the difference matters:
counting *registered* studies for a compound (which says something about
research activity) and retrieving *posted results* (which is actual evidence).
A registered trial with no results is not evidence of an effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from atlas.core.evidence import SourceRecord, Species, StudyDesign
from atlas.sources.base import CachedHttpClient

logger = logging.getLogger(__name__)

__all__ = ["ClinicalTrialsClient", "TrialRecord", "LICENSE_NOTE"]

API = "https://clinicaltrials.gov/api/v2"

LICENSE_NOTE = (
    "ClinicalTrials.gov data retrieved via API v2. Records are provided by study "
    "sponsors and investigators; the NLM does not verify their scientific "
    "content. Registry data are in the public domain."
)

_DESIGN_BY_ALLOCATION = {
    "RANDOMIZED": StudyDesign.RCT,
    "NON_RANDOMIZED": StudyDesign.CONTROLLED_NONRANDOM,
}


@dataclass(frozen=True, slots=True)
class TrialRecord:
    nct_id: str
    title: str | None
    overall_status: str | None
    phases: tuple[str, ...]
    study_type: str | None
    allocation: str | None
    enrollment: int | None
    enrollment_type: str | None
    conditions: tuple[str, ...]
    interventions: tuple[str, ...]
    primary_outcomes: tuple[str, ...]
    has_results: bool
    start_date: str | None
    completion_date: str | None
    sponsor: str | None
    source: SourceRecord

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"

    @property
    def design(self) -> StudyDesign:
        if self.study_type == "OBSERVATIONAL":
            return StudyDesign.OBSERVATIONAL_COHORT
        if self.allocation and self.allocation in _DESIGN_BY_ALLOCATION:
            return _DESIGN_BY_ALLOCATION[self.allocation]
        if self.study_type == "INTERVENTIONAL":
            return StudyDesign.SINGLE_ARM
        return StudyDesign.UNSPECIFIED

    @property
    def is_evidence_bearing(self) -> bool:
        """True only when results have actually been posted to the registry."""
        return self.has_results


class ClinicalTrialsClient:
    def __init__(self, client: CachedHttpClient | None = None):
        self.http = client or CachedHttpClient()

    def search(
        self,
        term: str,
        *,
        page_size: int = 50,
        with_results_only: bool = False,
        exact_phrase: bool = True,
    ) -> tuple[list[TrialRecord], int]:
        """Search studies by term.

        ``exact_phrase`` quotes the term, which matters more than it sounds.
        The registry tokenises an unquoted query, so "Amino H2O" matches 23
        studies on the token "amino" -- among them an infant respiratory
        infection trial and an ALS trial -- producing phantom evidence for a
        compound with none. Quoted, it correctly returns zero.

        Returns ``(records, total_count)``; the total is the registry's own
        count of matching studies, independent of the page returned.
        """
        query_term = term
        if exact_phrase and not (term.startswith('"') and term.endswith('"')):
            query_term = f'"{term}"'

        params: dict[str, Any] = {
            "query.term": query_term,
            "pageSize": min(page_size, 1000),
            "countTotal": "true",
            "format": "json",
        }
        if with_results_only:
            params["filter.advanced"] = "AREA[HasResults]true"

        payload, fetch = self.http.get_json(f"{API}/studies", params)
        total = int(payload.get("totalCount", 0))
        records: list[TrialRecord] = []

        for study in payload.get("studies", []):
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design_mod = protocol.get("designModule", {})
            arms = protocol.get("armsInterventionsModule", {})
            outcomes = protocol.get("outcomesModule", {})
            conditions_mod = protocol.get("conditionsModule", {})
            sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})

            nct_id = ident.get("nctId", "")
            if not nct_id:
                continue

            enrollment_info = design_mod.get("enrollmentInfo", {}) or {}
            design_info = design_mod.get("designInfo", {}) or {}

            source = SourceRecord(
                source_db="clinicaltrials.gov",
                source_id=nct_id,
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                title=ident.get("briefTitle"),
                retrieved_at=fetch.fetched_at,
                retrieval_endpoint=fetch.url,
                license_note=LICENSE_NOTE,
                content_sha256=SourceRecord.hash_payload(study),
            )

            records.append(
                TrialRecord(
                    nct_id=nct_id,
                    title=ident.get("briefTitle"),
                    overall_status=status.get("overallStatus"),
                    phases=tuple(design_mod.get("phases", []) or ()),
                    study_type=design_mod.get("studyType"),
                    allocation=design_info.get("allocation"),
                    enrollment=enrollment_info.get("count"),
                    enrollment_type=enrollment_info.get("type"),
                    conditions=tuple(conditions_mod.get("conditions", []) or ()),
                    interventions=tuple(
                        i.get("name", "")
                        for i in (arms.get("interventions") or [])
                        if i.get("name")
                    ),
                    primary_outcomes=tuple(
                        o.get("measure", "")
                        for o in (outcomes.get("primaryOutcomes") or [])
                        if o.get("measure")
                    ),
                    has_results=bool(study.get("hasResults")),
                    start_date=(status.get("startDateStruct") or {}).get("date"),
                    completion_date=(
                        status.get("primaryCompletionDateStruct") or {}
                    ).get("date"),
                    sponsor=(sponsor_mod.get("leadSponsor") or {}).get("name"),
                    source=source,
                )
            )
        return records, total

    def count(self, term: str, *, exact_phrase: bool = True) -> int:
        """Total registered studies matching ``term``, fetching one record."""
        _, total = self.search(term, page_size=1, exact_phrase=exact_phrase)
        return total

    def count_with_results(self, term: str, *, exact_phrase: bool = True) -> int:
        """Registered studies that have posted results.

        Posted results are evidence; a registered study without them is only
        evidence of research activity.
        """
        _, total = self.search(
            term, page_size=1, with_results_only=True, exact_phrase=exact_phrase
        )
        return total
