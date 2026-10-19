"""ChEMBL access for compound identity, development phase and bioactivity.

``max_phase`` is the field that matters most for this registry: it is ChEMBL's
record of how far a molecule has advanced in clinical development, and it is the
honest way to separate an approved drug from a research chemical without
asserting anything from memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from atlas.core.evidence import SourceRecord
from atlas.sources.base import CachedHttpClient, Fetch

logger = logging.getLogger(__name__)

__all__ = ["ChemblClient", "ChemblMolecule", "ChemblActivity", "LICENSE_NOTE"]

API = "https://www.ebi.ac.uk/chembl/api/data"

LICENSE_NOTE = (
    "ChEMBL data retrieved from EMBL-EBI. ChEMBL is released under "
    "Creative Commons Attribution-ShareAlike 3.0 Unported; attribution and "
    "share-alike terms apply to redistribution."
)

# ChEMBL's max_phase encoding. 4 = approved. None/0 means no recorded clinical
# development, which is a statement about the ChEMBL record, not proof that no
# study exists anywhere.
_PHASE_LABEL = {
    4.0: "approved",
    3.0: "phase_3",
    2.0: "phase_2",
    1.0: "phase_1",
    0.5: "early_phase_1",
    0.0: "no_recorded_clinical_phase",
}


@dataclass(frozen=True, slots=True)
class ChemblMolecule:
    chembl_id: str
    pref_name: str | None
    molecule_type: str | None
    max_phase: float | None
    full_mwt: float | None
    smiles: str | None
    inchikey: str | None
    first_approval: int | None
    withdrawn: bool | None
    synonyms: tuple[str, ...]
    source: SourceRecord

    @property
    def phase_label(self) -> str:
        if self.max_phase is None:
            return "no_recorded_clinical_phase"
        return _PHASE_LABEL.get(float(self.max_phase), f"phase_{self.max_phase:g}")

    @property
    def is_approved(self) -> bool:
        return self.max_phase is not None and float(self.max_phase) >= 4.0


@dataclass(frozen=True, slots=True)
class ChemblActivity:
    """One measured activity value against one target."""

    activity_id: int
    molecule_chembl_id: str
    target_chembl_id: str | None
    target_name: str | None
    target_organism: str | None
    standard_type: str | None
    standard_value: float | None
    standard_units: str | None
    standard_relation: str | None
    assay_description: str | None
    assay_type: str | None
    document_chembl_id: str | None
    pubmed_id: str | None
    doi: str | None

    @property
    def is_exact(self) -> bool:
        """``>`` or ``<`` relations are censored values, not point estimates."""
        return self.standard_relation == "="


def _as_float(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class ChemblClient:
    def __init__(self, client: CachedHttpClient | None = None):
        self.http = client or CachedHttpClient()

    def _molecule_from_payload(self, mol: dict[str, Any], fetch: Fetch) -> ChemblMolecule:
        props = mol.get("molecule_properties") or {}
        structs = mol.get("molecule_structures") or {}
        syns = tuple(
            s.get("molecule_synonym")
            for s in (mol.get("molecule_synonyms") or [])
            if s.get("molecule_synonym")
        )
        chembl_id = mol.get("molecule_chembl_id", "")
        source = SourceRecord(
            source_db="chembl",
            source_id=chembl_id,
            url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/",
            title=mol.get("pref_name") or chembl_id,
            retrieved_at=fetch.fetched_at,
            retrieval_endpoint=fetch.url,
            license_note=LICENSE_NOTE,
            content_sha256=SourceRecord.hash_payload(mol),
        )
        withdrawn = mol.get("withdrawn_flag")
        return ChemblMolecule(
            chembl_id=chembl_id,
            pref_name=mol.get("pref_name"),
            molecule_type=mol.get("molecule_type"),
            max_phase=_as_float(mol.get("max_phase")),
            full_mwt=_as_float(props.get("full_mwt")),
            smiles=structs.get("canonical_smiles"),
            inchikey=structs.get("standard_inchi_key"),
            first_approval=mol.get("first_approval"),
            withdrawn=bool(withdrawn) if withdrawn is not None else None,
            synonyms=syns,
            source=source,
        )

    def search_molecule(self, name: str, *, limit: int = 5) -> list[ChemblMolecule]:
        """Full-text molecule search. Ordered by ChEMBL relevance."""
        payload, fetch = self.http.get_json(
            f"{API}/molecule/search",
            {"q": name, "format": "json", "limit": limit},
        )
        return [
            self._molecule_from_payload(m, fetch) for m in payload.get("molecules", [])
        ]

    def molecule_by_id(self, chembl_id: str) -> ChemblMolecule | None:
        fetch = self.http.get(
            f"{API}/molecule/{chembl_id}", {"format": "json"}, accept="application/json"
        )
        if fetch.status == 404:
            return None
        return self._molecule_from_payload(fetch.json(), fetch)

    def activities(
        self,
        chembl_id: str,
        *,
        limit: int = 100,
        standard_types: tuple[str, ...] = ("Kd", "Ki", "IC50", "EC50"),
    ) -> list[ChemblActivity]:
        """Fetch measured activities, restricted to interpretable affinity types."""
        out: list[ChemblActivity] = []
        for std_type in standard_types:
            payload, _ = self.http.get_json(
                f"{API}/activity",
                {
                    "molecule_chembl_id": chembl_id,
                    "standard_type": std_type,
                    "format": "json",
                    "limit": limit,
                },
            )
            for act in payload.get("activities", []):
                out.append(
                    ChemblActivity(
                        activity_id=act.get("activity_id", -1),
                        molecule_chembl_id=act.get("molecule_chembl_id", chembl_id),
                        target_chembl_id=act.get("target_chembl_id"),
                        target_name=act.get("target_pref_name"),
                        target_organism=act.get("target_organism"),
                        standard_type=act.get("standard_type"),
                        standard_value=_as_float(act.get("standard_value")),
                        standard_units=act.get("standard_units"),
                        standard_relation=act.get("standard_relation"),
                        assay_description=act.get("assay_description"),
                        assay_type=act.get("assay_type"),
                        document_chembl_id=act.get("document_chembl_id"),
                        pubmed_id=str(act["document_pubmed_id"])
                        if act.get("document_pubmed_id")
                        else None,
                        doi=act.get("document_doi"),
                    )
                )
        return out
