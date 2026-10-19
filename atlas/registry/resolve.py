"""Molecular identity resolution with field-level provenance.

Two findings from live API behaviour shape this module:

1. **Relevance ranking is not identity.** ChEMBL's ``molecule/search`` returns
   the approved SEMAGLUTIDE record (CHEMBL2108724) *fourth*, behind three
   unnamed structure records that share its molecular weight. Taking the top
   hit assigns the wrong development status. Candidates are therefore scored on
   exact name agreement and recorded clinical phase, never on search rank alone.

2. **No single source holds a complete identity.** The named, approved ChEMBL
   record for semaglutide carries no computed molecular weight, while the
   unnamed structure records do. Each field consequently carries its own source,
   rather than the whole identity being attributed to one database.

An unresolved or ambiguous identity is a first-class outcome, because a
structure-dependent prediction must be blocked when identity is unknown.
"""

from __future__ import annotations

import enum
import logging
import re
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from atlas.core.evidence import SourceRecord
from atlas.sources.chembl import ChemblClient, ChemblMolecule
from atlas.sources.pubchem import PubChemClient, PubChemCompound

logger = logging.getLogger(__name__)

__all__ = [
    "IdentityStatus",
    "Modality",
    "ResolvedField",
    "MolecularIdentity",
    "IdentityResolver",
    "normalize_name",
]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Minimum candidate score that counts as agreement on identity. Scores below
# this mean no name or synonym matched exactly, so the candidate is rejected.
NAME_MATCH_FLOOR = 60.0


def normalize_name(name: str) -> str:
    """Casefold and strip punctuation so ``5-Amino-1MQ`` == ``5 amino 1mq``."""
    return _NON_ALNUM.sub("", name.casefold())


class IdentityStatus(str, enum.Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous_multiple_candidates"
    UNRESOLVED = "unresolved_no_database_match"
    MIXTURE = "mixture_not_a_single_molecular_entity"
    SUPPLIER_DEFINED = "supplier_defined_identity_unverified"

    @property
    def permits_structure_dependent_prediction(self) -> bool:
        """Only a resolved single entity may feed a structure-based model."""
        return self is IdentityStatus.RESOLVED


class Modality(str, enum.Enum):
    """Chemical class. Determines which featurization pipelines may apply."""

    PEPTIDE = "peptide"
    MODIFIED_PEPTIDE = "modified_peptide"
    PEPTIDE_HORMONE_ANALOG = "peptide_hormone_analog"
    PROTEIN = "protein"
    SMALL_MOLECULE = "small_molecule"
    NUCLEOTIDE_COFACTOR = "nucleotide_cofactor"
    METAL_COMPLEX = "metal_coordination_complex"
    MIXTURE = "mixture_or_formulation"
    EXCIPIENT_OR_DILUENT = "excipient_or_diluent"
    UNKNOWN = "unknown"

    @property
    def is_peptidic(self) -> bool:
        return self in (
            Modality.PEPTIDE,
            Modality.MODIFIED_PEPTIDE,
            Modality.PEPTIDE_HORMONE_ANALOG,
            Modality.PROTEIN,
        )


class ResolvedField(BaseModel):
    """One identity field together with the source it came from."""

    model_config = ConfigDict(frozen=True)

    value: Any
    source_db: str
    source_id: str
    url: str | None = None

    @property
    def source_key(self) -> str:
        return f"{self.source_db}:{self.source_id}"


class MolecularIdentity(BaseModel):
    """The resolved chemical identity of a registry entry."""

    model_config = ConfigDict(frozen=True)

    registry_key: str
    query_name: str
    status: IdentityStatus
    modality: Modality = Modality.UNKNOWN

    preferred_name: ResolvedField | None = None
    chembl_id: ResolvedField | None = None
    pubchem_cid: ResolvedField | None = None
    molecular_weight_g_per_mol: ResolvedField | None = None
    molecular_formula: ResolvedField | None = None
    smiles: ResolvedField | None = None
    inchikey: ResolvedField | None = None
    molecule_type: ResolvedField | None = None
    max_clinical_phase: ResolvedField | None = None
    first_approval_year: ResolvedField | None = None

    synonyms: tuple[str, ...] = ()
    candidates_considered: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    sources: tuple[SourceRecord, ...] = ()

    @property
    def mw(self) -> float | None:
        v = self.molecular_weight_g_per_mol
        return float(v.value) if v is not None and v.value is not None else None

    @property
    def is_approved_drug(self) -> bool:
        """Whether a *confidently identified* molecule is approved.

        Gated on identity status: a supplier-defined product or a mixture
        cannot inherit the approval status of whatever record a search engine
        happened to return for its trade name.
        """
        if self.status not in (IdentityStatus.RESOLVED, IdentityStatus.AMBIGUOUS):
            return False
        phase = self.max_clinical_phase
        if phase is None or phase.value is None:
            return False
        return float(phase.value) >= 4.0

    @property
    def completeness(self) -> float:
        """Fraction of core identity fields that were resolved."""
        core = (
            self.preferred_name,
            self.molecular_weight_g_per_mol,
            self.molecular_formula,
            self.smiles,
            self.inchikey,
            self.max_clinical_phase,
        )
        return sum(1 for f in core if f is not None and f.value is not None) / len(core)

    def blocks_structure_models(self) -> bool:
        return not (
            self.status.permits_structure_dependent_prediction and self.smiles is not None
        )


def _score_chembl_candidate(
    mol: ChemblMolecule, wanted: set[str]
) -> tuple[float, list[str]]:
    """Score how well a ChEMBL record matches the requested name.

    Name agreement dominates; recorded clinical phase breaks ties so an
    approved, named record outranks an unnamed structure stub.
    """
    reasons: list[str] = []
    score = 0.0

    pref_norm = normalize_name(mol.pref_name) if mol.pref_name else ""
    if pref_norm and pref_norm in wanted:
        score += 100.0
        reasons.append("exact preferred-name match")
    else:
        syn_norms = {normalize_name(s) for s in mol.synonyms if s}
        if syn_norms & wanted:
            score += 60.0
            reasons.append("exact synonym match")

    if mol.max_phase is not None:
        score += 2.0 * float(mol.max_phase)
        if float(mol.max_phase) >= 4.0:
            reasons.append("approved in ChEMBL")

    if mol.molecule_type and mol.molecule_type != "Unknown":
        score += 5.0
    if mol.smiles:
        score += 3.0
    if mol.pref_name:
        score += 2.0

    return score, reasons


class IdentityResolver:
    """Resolves a name plus aliases into a :class:`MolecularIdentity`."""

    def __init__(
        self,
        chembl: ChemblClient | None = None,
        pubchem: PubChemClient | None = None,
    ) -> None:
        self.chembl = chembl or ChemblClient()
        self.pubchem = pubchem or PubChemClient()

    def resolve(
        self,
        registry_key: str,
        query_name: str,
        *,
        aliases: Iterable[str] = (),
        declared_modality: Modality = Modality.UNKNOWN,
        is_mixture: bool = False,
        supplier_defined: bool = False,
        search_limit: int = 8,
    ) -> MolecularIdentity:
        names = [query_name, *aliases]
        wanted = {normalize_name(n) for n in names if n}
        caveats: list[str] = []
        sources: list[SourceRecord] = []

        if is_mixture:
            return MolecularIdentity(
                registry_key=registry_key,
                query_name=query_name,
                status=IdentityStatus.MIXTURE,
                modality=Modality.MIXTURE,
                synonyms=tuple(aliases),
                caveats=(
                    "Entry is a formulation, not a single molecular entity. "
                    "Component identities and ratios must be verified per product; "
                    "constituent evidence does not transfer to the mixture.",
                ),
            )

        # -- ChEMBL: identity and development status ------------------------
        best: ChemblMolecule | None = None
        best_score = 0.0
        runner_up_score = 0.0
        considered: list[str] = []
        match_reasons: list[str] = []

        name_matched: dict[str, ChemblMolecule] = {}

        for name in names:
            try:
                candidates = self.chembl.search_molecule(name, limit=search_limit)
            except Exception as exc:  # network/parse issues must not abort resolution
                logger.warning("ChEMBL search failed for %r: %s", name, exc)
                caveats.append(f"ChEMBL search failed for {name!r}: {exc}")
                continue
            for mol in candidates:
                considered.append(mol.chembl_id)
                score, reasons = _score_chembl_candidate(mol, wanted)
                if score >= NAME_MATCH_FLOOR:
                    name_matched[mol.chembl_id] = mol
                if score > best_score:
                    runner_up_score = best_score
                    best, best_score, match_reasons = mol, score, reasons
                elif score > runner_up_score:
                    runner_up_score = score

        # Hard gate: without exact agreement on a name or synonym, a candidate is
        # rejected outright rather than adopted with a warning. Searching
        # "5-Amino-1MQ" returns 5-aminovaleric acid as ChEMBL's top relevance
        # hit; adopting it would assign an unrelated molecule's structure,
        # weight and development status to this registry entry.
        if best is not None and best_score < NAME_MATCH_FLOOR:
            caveats.append(
                f"ChEMBL returned {best.chembl_id} "
                f"({best.pref_name or 'unnamed'}) as its best relevance hit, but no "
                "name or synonym matched exactly; the candidate was rejected rather "
                "than adopted."
            )
            best = None
            best_score = 0.0
            runner_up_score = 0.0
            name_matched.clear()

        # Several records sharing one preferred name are usually duplicate
        # deposits of the same molecule (parent/salt records), which is not an
        # identity problem. It becomes one only when the same-named records
        # describe structurally different molecules -- as with kisspeptin,
        # where ChEMBL holds both a ~1302 Da and a ~5857 Da entity under the
        # same name. Ambiguity is therefore decided on structure, not on count.
        structurally_ambiguous = False
        if len(name_matched) > 1:
            skeletons = {
                m.inchikey.split("-")[0] for m in name_matched.values() if m.inchikey
            }
            weights = [m.full_mwt for m in name_matched.values() if m.full_mwt]
            if len(skeletons) > 1:
                structurally_ambiguous = True
            elif weights and min(weights) > 0 and (
                (max(weights) - min(weights)) / max(weights) > 0.02
            ):
                structurally_ambiguous = True

            if structurally_ambiguous:
                caveats.append(
                    f"{len(name_matched)} ChEMBL records matched this name but "
                    "describe structurally different molecules "
                    f"({', '.join(sorted(name_matched))}); the name alone does not "
                    "identify which entity was studied."
                )
            else:
                caveats.append(
                    f"{len(name_matched)} ChEMBL records share this name and agree "
                    "structurally (duplicate deposits); the highest-phase record "
                    "was used."
                )

        # -- PubChem: structure and molecular weight ------------------------
        pubchem_hit: PubChemCompound | None = None
        pubchem_matched_alias: str | None = None
        for name in names:
            try:
                hit = self.pubchem.lookup(name)
            except Exception as exc:
                logger.warning("PubChem lookup failed for %r: %s", name, exc)
                caveats.append(f"PubChem lookup failed for {name!r}: {exc}")
                continue
            if hit is not None:
                pubchem_hit, pubchem_matched_alias = hit, name
                break

        # PubChem resolves through its own curated name index, so a hit is a
        # name match by construction. Record which alias produced it.
        if pubchem_hit is not None and pubchem_matched_alias != query_name:
            caveats.append(
                f"PubChem identity was resolved via the alias "
                f"{pubchem_matched_alias!r} rather than {query_name!r}."
            )

        if best is None and pubchem_hit is None:
            return MolecularIdentity(
                registry_key=registry_key,
                query_name=query_name,
                status=IdentityStatus.UNRESOLVED,
                modality=declared_modality,
                synonyms=tuple(aliases),
                candidates_considered=tuple(considered),
                caveats=(
                    *caveats,
                    "Neither ChEMBL nor PubChem returned a record for this name or "
                    "its aliases. Structure-dependent predictions are blocked.",
                ),
            )

        fields: dict[str, ResolvedField | None] = {}

        if best is not None:
            sources.append(best.source)
            ck = dict(source_db="chembl", source_id=best.chembl_id, url=best.source.url)
            if best.pref_name:
                fields["preferred_name"] = ResolvedField(value=best.pref_name, **ck)
            fields["chembl_id"] = ResolvedField(value=best.chembl_id, **ck)
            if best.molecule_type:
                fields["molecule_type"] = ResolvedField(value=best.molecule_type, **ck)
            if best.max_phase is not None:
                fields["max_clinical_phase"] = ResolvedField(value=best.max_phase, **ck)
            if best.first_approval:
                fields["first_approval_year"] = ResolvedField(
                    value=best.first_approval, **ck
                )
            if best.full_mwt is not None:
                fields["molecular_weight_g_per_mol"] = ResolvedField(
                    value=best.full_mwt, **ck
                )
            if best.smiles:
                fields["smiles"] = ResolvedField(value=best.smiles, **ck)
            if best.inchikey:
                fields["inchikey"] = ResolvedField(value=best.inchikey, **ck)
            if best.withdrawn:
                caveats.append("ChEMBL records this molecule as withdrawn.")

        if pubchem_hit is not None:
            sources.append(pubchem_hit.source)
            pk = dict(
                source_db="pubchem",
                source_id=str(pubchem_hit.cid),
                url=pubchem_hit.url,
            )
            fields["pubchem_cid"] = ResolvedField(value=pubchem_hit.cid, **pk)
            # PubChem fills the gaps ChEMBL's named protein records leave open.
            if fields.get("molecular_weight_g_per_mol") is None and (
                pubchem_hit.molecular_weight is not None
            ):
                fields["molecular_weight_g_per_mol"] = ResolvedField(
                    value=pubchem_hit.molecular_weight, **pk
                )
            if pubchem_hit.molecular_formula:
                fields["molecular_formula"] = ResolvedField(
                    value=pubchem_hit.molecular_formula, **pk
                )
            if fields.get("smiles") is None and pubchem_hit.smiles:
                fields["smiles"] = ResolvedField(value=pubchem_hit.smiles, **pk)
            if fields.get("inchikey") is None and pubchem_hit.inchikey:
                fields["inchikey"] = ResolvedField(value=pubchem_hit.inchikey, **pk)
            if fields.get("preferred_name") is None and pubchem_hit.iupac_name:
                fields["preferred_name"] = ResolvedField(
                    value=pubchem_hit.iupac_name, **pk
                )

        # Cross-source molecular weight disagreement is a real identity signal:
        # it usually means the two databases hold different molecular forms.
        if (
            best is not None
            and best.full_mwt is not None
            and pubchem_hit is not None
            and pubchem_hit.molecular_weight is not None
        ):
            a, b = float(best.full_mwt), float(pubchem_hit.molecular_weight)
            if abs(a - b) / max(a, b) > 0.02:
                # Two databases describing different-sized molecules under one
                # name is identity ambiguity, not a footnote: kisspeptin
                # resolves to ~1302 Da in ChEMBL and ~5857 Da in PubChem
                # (kisspeptin-10 vs kisspeptin-54). Downgrade the status so
                # structure-dependent models stay blocked.
                structurally_ambiguous = True
                caveats.append(
                    f"Molecular weight disagrees between ChEMBL ({a:g}) and "
                    f"PubChem ({b:g}); the databases hold different molecular "
                    "forms, salts or conjugates under this name."
                )

        status = IdentityStatus.RESOLVED
        if supplier_defined:
            status = IdentityStatus.SUPPLIER_DEFINED
            caveats.append(
                "Identity is supplier-defined; composition must be verified "
                "against an analytical certificate before any pharmacological "
                "classification is applied."
            )
        elif structurally_ambiguous:
            status = IdentityStatus.AMBIGUOUS

        modality = declared_modality
        mtype = fields.get("molecule_type")
        if modality is Modality.UNKNOWN and mtype is not None:
            modality = {
                "Protein": Modality.PROTEIN,
                "Small molecule": Modality.SMALL_MOLECULE,
                "Oligosaccharide": Modality.UNKNOWN,
                "Oligonucleotide": Modality.UNKNOWN,
            }.get(str(mtype.value), Modality.UNKNOWN)

        syns: list[str] = list(aliases)
        if best is not None:
            syns.extend(best.synonyms)
        if pubchem_hit is not None:
            syns.extend(pubchem_hit.synonyms[:10])
        seen: set[str] = set()
        unique_syns = tuple(
            s for s in syns if s and not (normalize_name(s) in seen or seen.add(normalize_name(s)))
        )

        return MolecularIdentity(
            registry_key=registry_key,
            query_name=query_name,
            status=status,
            modality=modality,
            synonyms=unique_syns,
            candidates_considered=tuple(dict.fromkeys(considered)),
            caveats=tuple(caveats),
            sources=tuple(sources),
            **fields,
        )
