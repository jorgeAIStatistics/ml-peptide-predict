"""PubChem PUG REST access for small-molecule and cofactor identity.

Used for entries that are genuinely small molecules or nucleotide cofactors
rather than peptides. PubChem property names changed in 2025 (``CanonicalSMILES``
was split into ``SMILES`` and ``ConnectivitySMILES``), so the client probes the
modern names and falls back, instead of hard-coding one vintage of the API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from atlas.core.evidence import SourceRecord
from atlas.sources.base import CachedHttpClient, Fetch, FetchError

logger = logging.getLogger(__name__)

__all__ = ["PubChemClient", "PubChemCompound", "LICENSE_NOTE"]

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

LICENSE_NOTE = (
    "PubChem data retrieved via PUG REST. PubChem aggregates depositor-contributed "
    "records; individual depositor licences vary. NCBI itself places no "
    "restrictions on reuse of the aggregated data."
)

_PROPERTY_SETS = (
    "MolecularFormula,MolecularWeight,SMILES,InChIKey,IUPACName",
    "MolecularFormula,MolecularWeight,ConnectivitySMILES,InChIKey,IUPACName",
    "MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey,IUPACName",
)


@dataclass(frozen=True, slots=True)
class PubChemCompound:
    cid: int
    molecular_formula: str | None
    molecular_weight: float | None
    smiles: str | None
    inchikey: str | None
    iupac_name: str | None
    synonyms: tuple[str, ...]
    source: SourceRecord

    @property
    def url(self) -> str:
        return f"https://pubchem.ncbi.nlm.nih.gov/compound/{self.cid}"


class PubChemClient:
    def __init__(self, client: CachedHttpClient | None = None):
        self.http = client or CachedHttpClient()

    def cids_for_name(self, name: str) -> tuple[list[int], Fetch]:
        """Resolve a chemical name to PubChem CIDs. Empty list if unknown."""
        url = f"{PUG}/compound/name/{quote(name, safe='')}/cids/JSON"
        fetch = self.http.get(url, accept="application/json")
        if fetch.status == 404:
            return [], fetch
        payload = fetch.json()
        return list(payload.get("IdentifierList", {}).get("CID", [])), fetch

    def properties(self, cid: int) -> tuple[dict[str, Any], Fetch]:
        """Fetch core computed properties, tolerating API property renames."""
        last: FetchError | None = None
        for props in _PROPERTY_SETS:
            url = f"{PUG}/compound/cid/{cid}/property/{props}/JSON"
            try:
                fetch = self.http.get(url, accept="application/json")
            except FetchError as exc:
                last = exc
                continue
            if fetch.status == 404:
                last = FetchError("property set rejected", url=url, status=404)
                continue
            rows = fetch.json().get("PropertyTable", {}).get("Properties", [])
            if rows:
                return rows[0], fetch
        if last is not None:
            raise last
        return {}, fetch

    def synonyms(self, cid: int, *, limit: int = 40) -> tuple[list[str], Fetch]:
        url = f"{PUG}/compound/cid/{cid}/synonyms/JSON"
        fetch = self.http.get(url, accept="application/json")
        if fetch.status == 404:
            return [], fetch
        info = fetch.json().get("InformationList", {}).get("Information", [])
        if not info:
            return [], fetch
        return list(info[0].get("Synonym", []))[:limit], fetch

    def lookup(self, name: str) -> PubChemCompound | None:
        """Resolve a name to a single compound, or ``None`` when unresolvable.

        When a name maps to several CIDs the lowest is taken, which is PubChem's
        own relevance ordering, and the ambiguity is logged — the registry
        records multi-CID hits as an identity caveat rather than hiding them.
        """
        cids, search_fetch = self.cids_for_name(name)
        if not cids:
            return None
        if len(cids) > 1:
            logger.info("name %r matched %d PubChem CIDs; taking %d", name, len(cids), cids[0])
        cid = cids[0]
        props, prop_fetch = self.properties(cid)
        syns, _ = self.synonyms(cid)

        smiles = props.get("SMILES") or props.get("ConnectivitySMILES") or props.get(
            "CanonicalSMILES"
        )
        mw = props.get("MolecularWeight")
        source = SourceRecord(
            source_db="pubchem",
            source_id=str(cid),
            url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            title=props.get("IUPACName") or name,
            retrieved_at=prop_fetch.fetched_at,
            retrieval_endpoint=prop_fetch.url,
            license_note=LICENSE_NOTE,
            content_sha256=SourceRecord.hash_payload(props),
        )
        return PubChemCompound(
            cid=cid,
            molecular_formula=props.get("MolecularFormula"),
            molecular_weight=float(mw) if mw not in (None, "") else None,
            smiles=smiles,
            inchikey=props.get("InChIKey"),
            iupac_name=props.get("IUPACName"),
            synonyms=tuple(syns),
            source=source,
        )
