"""PubMed access via NCBI E-utilities.

Returns :class:`SourceRecord` objects with real identifiers, publication types
mapped to :class:`StudyDesign`, and retraction status actually checked rather
than assumed. Abstract text is returned separately from the citation record
because abstracts carry publisher copyright while the citation metadata does
not; callers decide what they may store or display.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from atlas.core.evidence import SourceRecord, StudyDesign
from atlas.sources.base import CachedHttpClient, Fetch

logger = logging.getLogger(__name__)

__all__ = ["PubMedClient", "PubMedRecord", "map_publication_types"]

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

LICENSE_NOTE = (
    "PubMed citation metadata retrieved via NCBI E-utilities. NLM terms permit "
    "retrieval and use of citation data; abstract text remains subject to "
    "publisher copyright. Verify terms before redistribution."
)

# PubMed publication types that identify a study design, strongest first. A
# record often carries several; the first match in this order wins.
_DESIGN_BY_PUBTYPE: list[tuple[str, StudyDesign]] = [
    ("Meta-Analysis", StudyDesign.REVIEW_SYSTEMATIC),
    ("Systematic Review", StudyDesign.REVIEW_SYSTEMATIC),
    ("Randomized Controlled Trial", StudyDesign.RCT),
    ("Controlled Clinical Trial", StudyDesign.CONTROLLED_NONRANDOM),
    ("Clinical Trial, Phase I", StudyDesign.SINGLE_ARM),
    ("Clinical Trial, Phase II", StudyDesign.SINGLE_ARM),
    ("Clinical Trial, Phase III", StudyDesign.RCT),
    ("Clinical Trial", StudyDesign.SINGLE_ARM),
    ("Observational Study", StudyDesign.OBSERVATIONAL_COHORT),
    ("Case Reports", StudyDesign.CASE_REPORT),
    ("Review", StudyDesign.REVIEW_NARRATIVE),
]

_RETRACTION_TYPES = {"Retracted Publication", "Retraction of Publication"}

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        ],
        start=1,
    )
}


def map_publication_types(pub_types: Iterable[str]) -> StudyDesign:
    """Map PubMed publication types to a study design.

    Returns ``UNSPECIFIED`` when no type is informative about design, which is
    the common case for basic-science papers — those need the design read from
    the methods, not guessed from metadata.
    """
    types = set(pub_types)
    for name, design in _DESIGN_BY_PUBTYPE:
        if name in types:
            return design
    return StudyDesign.UNSPECIFIED


@dataclass(frozen=True, slots=True)
class PubMedRecord:
    """A PubMed citation plus the text fields needed for extraction."""

    source: SourceRecord
    abstract: str | None
    design: StudyDesign
    mesh_terms: tuple[str, ...] = ()

    @property
    def pmid(self) -> str:
        return self.source.source_id


def _text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = "".join(node.itertext()).strip()
    return text or None


def _parse_pubdate(article: ET.Element) -> date | None:
    """Extract a publication date, preferring the electronic article date."""
    for path in ("ArticleDate", "Journal/JournalIssue/PubDate"):
        node = article.find(path)
        if node is None:
            continue
        year_text = _text(node.find("Year"))
        if not year_text:
            # PubDate sometimes carries only a MedlineDate like "2019 Nov-Dec".
            medline = _text(node.find("MedlineDate"))
            if medline:
                match = re.search(r"(\d{4})", medline)
                if match:
                    return date(int(match.group(1)), 1, 1)
            continue
        year = int(year_text)
        month_text = (_text(node.find("Month")) or "1").strip()
        if month_text.isdigit():
            month = int(month_text)
        else:
            month = _MONTHS.get(month_text[:3].lower(), 1)
        day_text = _text(node.find("Day")) or "1"
        day = int(day_text) if day_text.isdigit() else 1
        try:
            return date(year, max(1, min(12, month)), max(1, min(28, day)))
        except ValueError:
            return date(year, 1, 1)
    return None


def _parse_authors(article: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in article.findall("AuthorList/Author"):
        last = _text(author.find("LastName"))
        initials = _text(author.find("Initials"))
        collective = _text(author.find("CollectiveName"))
        if last:
            authors.append(f"{last} {initials}" if initials else last)
        elif collective:
            authors.append(collective)
    return authors


def _parse_abstract(article: ET.Element) -> str | None:
    chunks: list[str] = []
    for node in article.findall("Abstract/AbstractText"):
        label = node.get("Label")
        body = _text(node)
        if not body:
            continue
        chunks.append(f"{label}: {body}" if label else body)
    return "\n\n".join(chunks) if chunks else None


class PubMedClient:
    """Search and fetch PubMed records."""

    def __init__(self, client: CachedHttpClient | None = None, *, api_key: str | None = None):
        self.http = client or CachedHttpClient()
        self.api_key = api_key

    def _common(self) -> dict[str, Any]:
        params: dict[str, Any] = {"db": "pubmed", "tool": "atlas", "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(
        self, query: str, *, retmax: int = 50, mindate: str | None = None
    ) -> tuple[list[str], int, Fetch]:
        """Run an ESearch query.

        Returns ``(pmids, total_hits, fetch)``. ``total_hits`` is the full
        result count, which is often more informative than the returned page:
        a query with two hits is itself evidence about how sparse a literature is.
        """
        params = self._common() | {"term": query, "retmax": retmax, "sort": "relevance"}
        if mindate:
            params |= {"mindate": mindate, "datetype": "pdat"}
        fetch = self.http.get(f"{EUTILS}/esearch.fcgi", params)
        root = ET.fromstring(fetch.body)
        count_node = root.find("Count")
        total = int(count_node.text) if count_node is not None and count_node.text else 0
        pmids = [el.text for el in root.findall("IdList/Id") if el.text]
        return pmids, total, fetch

    def fetch(self, pmids: list[str]) -> tuple[list[PubMedRecord], Fetch | None]:
        """Fetch full citation records for a list of PMIDs."""
        if not pmids:
            return [], None
        params = self._common() | {"id": ",".join(pmids)}
        fetch = self.http.get(f"{EUTILS}/efetch.fcgi", params)
        root = ET.fromstring(fetch.body)
        records: list[PubMedRecord] = []

        for entry in root.findall("PubmedArticle"):
            citation = entry.find("MedlineCitation")
            if citation is None:
                continue
            pmid = _text(citation.find("PMID"))
            article = citation.find("Article")
            if pmid is None or article is None:
                continue

            ids = {
                el.get("IdType"): (el.text or "").strip()
                for el in entry.findall("PubmedData/ArticleIdList/ArticleId")
            }
            pub_types = [
                t
                for t in (
                    _text(el) for el in article.findall("PublicationTypeList/PublicationType")
                )
                if t
            ]
            journal = _text(article.find("Journal/ISOAbbreviation")) or _text(
                article.find("Journal/Title")
            )
            mesh = tuple(
                t
                for t in (
                    _text(el)
                    for el in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
                )
                if t
            )

            payload = {
                "pmid": pmid,
                "title": _text(article.find("ArticleTitle")),
                "pub_types": sorted(pub_types),
                "ids": ids,
            }

            source = SourceRecord(
                source_db="pubmed",
                source_id=pmid,
                pmid=pmid,
                doi=ids.get("doi") or None,
                pmcid=ids.get("pmc") or None,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                title=_text(article.find("ArticleTitle")),
                journal=journal,
                authors=_parse_authors(article),
                publication_date=_parse_pubdate(article),
                publication_types=sorted(pub_types),
                retrieved_at=fetch.fetched_at,
                retrieval_endpoint=fetch.url,
                license_note=LICENSE_NOTE,
                content_sha256=SourceRecord.hash_payload(payload),
                # Retraction status was genuinely inspected here, so False is a
                # real finding rather than an untested default.
                is_retracted=bool(_RETRACTION_TYPES & set(pub_types)),
            )
            records.append(
                PubMedRecord(
                    source=source,
                    abstract=_parse_abstract(article),
                    design=map_publication_types(pub_types),
                    mesh_terms=mesh,
                )
            )
        return records, fetch

    def search_and_fetch(
        self, query: str, *, retmax: int = 20
    ) -> tuple[list[PubMedRecord], int]:
        """Convenience: search then fetch. Returns ``(records, total_hits)``."""
        pmids, total, _ = self.search(query, retmax=retmax)
        records, _ = self.fetch(pmids)
        return records, total
