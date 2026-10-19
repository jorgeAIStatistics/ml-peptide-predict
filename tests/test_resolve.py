"""Identity-resolution tests.

Fixtures are the real payloads observed from ChEMBL and PubChem, so these run
offline and pin the behaviour that live testing exposed: relevance rank is not
identity, and a non-matching name must be rejected rather than adopted.
"""

from __future__ import annotations

from atlas.core.evidence import SourceRecord
from atlas.registry.resolve import (
    IdentityResolver,
    IdentityStatus,
    Modality,
    normalize_name,
)
from atlas.sources.chembl import ChemblMolecule
from atlas.sources.pubchem import PubChemCompound


def _src(db: str, sid: str) -> SourceRecord:
    return SourceRecord(source_db=db, source_id=sid, url=f"https://example/{sid}")


def mol(
    chembl_id: str,
    pref_name: str | None,
    *,
    molecule_type: str | None = "Protein",
    max_phase: float | None = None,
    full_mwt: float | None = None,
    smiles: str | None = None,
    inchikey: str | None = None,
    synonyms: tuple[str, ...] = (),
    withdrawn: bool | None = None,
) -> ChemblMolecule:
    return ChemblMolecule(
        chembl_id=chembl_id,
        pref_name=pref_name,
        molecule_type=molecule_type,
        max_phase=max_phase,
        full_mwt=full_mwt,
        smiles=smiles,
        inchikey=inchikey,
        first_approval=None,
        withdrawn=withdrawn,
        synonyms=synonyms,
        source=_src("chembl", chembl_id),
    )


def compound(
    cid: int,
    *,
    mw: float | None = None,
    formula: str | None = None,
    smiles: str | None = "CC",
    iupac: str | None = None,
) -> PubChemCompound:
    return PubChemCompound(
        cid=cid,
        molecular_formula=formula,
        molecular_weight=mw,
        smiles=smiles,
        inchikey=None,
        iupac_name=iupac,
        synonyms=(),
        source=_src("pubchem", str(cid)),
    )


class StubChembl:
    def __init__(self, results: dict[str, list[ChemblMolecule]] | None = None):
        self.results = results or {}
        self.calls: list[str] = []

    def search_molecule(self, name: str, *, limit: int = 5) -> list[ChemblMolecule]:
        self.calls.append(name)
        return self.results.get(name, [])


class StubPubChem:
    def __init__(self, results: dict[str, PubChemCompound] | None = None):
        self.results = results or {}
        self.calls: list[str] = []

    def lookup(self, name: str) -> PubChemCompound | None:
        self.calls.append(name)
        return self.results.get(name)


def resolver(chembl: StubChembl | None = None, pubchem: StubPubChem | None = None):
    return IdentityResolver(chembl=chembl or StubChembl(), pubchem=pubchem or StubPubChem())


class TestNormalizeName:
    def test_punctuation_and_case_ignored(self):
        assert normalize_name("5-Amino-1MQ") == normalize_name("5 amino 1mq")
        assert normalize_name("CJC-1295") == "cjc1295"
        assert normalize_name("NAD+") == "nad"

    def test_distinct_names_stay_distinct(self):
        assert normalize_name("AHK-Cu") != normalize_name("GHK-Cu")


class TestRelevanceRankIsNotIdentity:
    """Regression test for the live ChEMBL ordering observed for semaglutide."""

    def _semaglutide_candidates(self) -> list[ChemblMolecule]:
        # Observed order: three unnamed structure records outrank the named,
        # approved record.
        return [
            mol("CHEMBL4556788", None, molecule_type="Unknown", full_mwt=4113.64),
            mol("CHEMBL4752331", None, molecule_type="Unknown", full_mwt=4113.64),
            mol("CHEMBL6147864", None, molecule_type=None, full_mwt=4113.64),
            mol("CHEMBL2108724", "SEMAGLUTIDE", molecule_type="Protein", max_phase=4.0),
        ]

    def test_named_approved_record_beats_higher_ranked_stubs(self):
        r = resolver(StubChembl({"semaglutide": self._semaglutide_candidates()}))
        ident = r.resolve("semaglutide", "semaglutide")
        assert ident.chembl_id is not None
        assert ident.chembl_id.value == "CHEMBL2108724"
        assert ident.is_approved_drug

    def test_all_candidates_are_recorded(self):
        r = resolver(StubChembl({"semaglutide": self._semaglutide_candidates()}))
        ident = r.resolve("semaglutide", "semaglutide")
        assert len(ident.candidates_considered) == 4

    def test_missing_weight_is_filled_from_pubchem_with_its_own_provenance(self):
        """The named ChEMBL record has no MW; PubChem supplies it."""
        r = resolver(
            StubChembl({"semaglutide": self._semaglutide_candidates()}),
            StubPubChem({"semaglutide": compound(56843331, mw=4114.0, formula="C187H291N45O59")}),
        )
        ident = r.resolve("semaglutide", "semaglutide")
        assert ident.mw == 4114.0
        assert ident.molecular_weight_g_per_mol.source_db == "pubchem"
        assert ident.chembl_id.value == "CHEMBL2108724"


class TestWeakMatchIsRejected:
    """Searching 5-Amino-1MQ returns 5-aminovaleric acid as ChEMBL's top hit."""

    def test_non_matching_candidate_is_not_adopted(self):
        r = resolver(
            StubChembl(
                {"5-Amino-1MQ": [mol("CHEMBL284116", "5-AMINOVALERIC ACID", full_mwt=117.15)]}
            )
        )
        ident = r.resolve("5-amino-1mq", "5-Amino-1MQ")
        assert ident.chembl_id is None
        assert ident.status is IdentityStatus.UNRESOLVED

    def test_rejection_is_explained_in_caveats(self):
        r = resolver(
            StubChembl(
                {"5-Amino-1MQ": [mol("CHEMBL284116", "5-AMINOVALERIC ACID", full_mwt=117.15)]}
            )
        )
        ident = r.resolve("5-amino-1mq", "5-Amino-1MQ")
        assert any("rejected rather than adopted" in c for c in ident.caveats)
        assert any("5-AMINOVALERIC ACID" in c for c in ident.caveats)

    def test_alias_resolves_through_pubchem_when_chembl_fails(self):
        r = resolver(
            StubChembl(
                {"5-Amino-1MQ": [mol("CHEMBL284116", "5-AMINOVALERIC ACID", full_mwt=117.15)]}
            ),
            StubPubChem(
                {
                    "5-amino-1-methylquinolinium": compound(
                        950107,
                        mw=159.21,
                        formula="C10H11N2+",
                        iupac="1-methylquinolin-1-ium-5-amine",
                    )
                }
            ),
        )
        ident = r.resolve(
            "5-amino-1mq", "5-Amino-1MQ", aliases=("5-amino-1-methylquinolinium",)
        )
        assert ident.status is IdentityStatus.RESOLVED
        assert ident.mw == 159.21
        assert ident.molecular_formula.value == "C10H11N2+"
        assert any("via the alias" in c for c in ident.caveats)

    def test_exact_synonym_match_is_accepted(self):
        r = resolver(
            StubChembl(
                {"NAD+": [mol("CHEMBL1234613", "NADIDE", synonyms=("NAD", "NAD+"))]}
            )
        )
        ident = r.resolve("nad", "NAD+")
        assert ident.chembl_id.value == "CHEMBL1234613"
        assert ident.status is IdentityStatus.RESOLVED


class TestMixtures:
    def test_mixture_short_circuits_before_any_lookup(self):
        chembl, pubchem = StubChembl(), StubPubChem()
        ident = resolver(chembl, pubchem).resolve(
            "glow", "GLOW blend", is_mixture=True
        )
        assert ident.status is IdentityStatus.MIXTURE
        assert ident.modality is Modality.MIXTURE
        # No database should be consulted for a formulation identity.
        assert chembl.calls == [] and pubchem.calls == []

    def test_mixture_blocks_structure_models(self):
        ident = resolver().resolve("klow", "KLOW blend", is_mixture=True)
        assert ident.blocks_structure_models()

    def test_mixture_is_never_approved(self):
        ident = resolver().resolve("glow", "GLOW blend", is_mixture=True)
        assert not ident.is_approved_drug

    def test_mixture_caveat_blocks_constituent_transfer(self):
        ident = resolver().resolve("glow", "GLOW blend", is_mixture=True)
        assert any("does not transfer" in c for c in ident.caveats)


class TestSupplierDefined:
    def test_supplier_product_does_not_inherit_approval(self):
        """A trade name must not absorb an approved record's status."""
        r = resolver(
            StubChembl({"Amino H2O": [mol("CHEMBL1201498", "AMINO ACIDS", max_phase=4.0)]}),
            StubPubChem({"Amino H2O": compound(1, mw=100.0)}),
        )
        ident = r.resolve("amino-h2o", "Amino H2O", supplier_defined=True)
        assert ident.status is IdentityStatus.SUPPLIER_DEFINED
        assert not ident.is_approved_drug

    def test_supplier_identity_requires_verification_caveat(self):
        r = resolver(pubchem=StubPubChem({"Amino H2O": compound(1, mw=100.0)}))
        ident = r.resolve("amino-h2o", "Amino H2O", supplier_defined=True)
        assert any("analytical certificate" in c for c in ident.caveats)

    def test_supplier_identity_blocks_structure_models(self):
        r = resolver(pubchem=StubPubChem({"Amino H2O": compound(1, smiles="CC")}))
        ident = r.resolve("amino-h2o", "Amino H2O", supplier_defined=True)
        assert ident.blocks_structure_models()


class TestUnresolved:
    def test_no_match_anywhere_is_unresolved(self):
        ident = resolver().resolve("mystery", "Nonexistent Compound XYZ")
        assert ident.status is IdentityStatus.UNRESOLVED
        assert ident.blocks_structure_models()
        assert ident.completeness == 0.0

    def test_unresolved_states_that_predictions_are_blocked(self):
        ident = resolver().resolve("mystery", "Nonexistent Compound XYZ")
        assert any("blocked" in c for c in ident.caveats)


class TestCrossSourceDisagreement:
    def test_divergent_molecular_weight_is_flagged(self):
        r = resolver(
            StubChembl({"thing": [mol("CHEMBL1", "THING", full_mwt=117.15)]}),
            StubPubChem({"thing": compound(2, mw=159.21)}),
        )
        ident = r.resolve("thing", "thing")
        assert any("disagrees between ChEMBL" in c for c in ident.caveats)

    def test_divergent_molecular_weight_downgrades_status(self):
        """Kisspeptin: ~1302 Da in ChEMBL vs ~5857 Da in PubChem."""
        r = resolver(
            StubChembl({"kisspeptin": [mol("CHEMBL_K", "KISSPEPTIN", full_mwt=1302.46)]}),
            StubPubChem({"kisspeptin": compound(2, mw=5857.0)}),
        )
        ident = r.resolve("kisspeptin", "kisspeptin")
        assert ident.status is IdentityStatus.AMBIGUOUS
        assert ident.blocks_structure_models()

    def test_agreeing_weights_are_not_flagged(self):
        r = resolver(
            StubChembl({"thing": [mol("CHEMBL1", "THING", full_mwt=1419.5)]}),
            StubPubChem({"thing": compound(2, mw=1419.56)}),
        )
        ident = r.resolve("thing", "thing")
        assert not any("disagrees" in c for c in ident.caveats)

    def test_chembl_weight_is_preferred_when_both_present(self):
        r = resolver(
            StubChembl({"thing": [mol("CHEMBL1", "THING", full_mwt=1419.5)]}),
            StubPubChem({"thing": compound(2, mw=1419.56)}),
        )
        ident = r.resolve("thing", "thing")
        assert ident.molecular_weight_g_per_mol.source_db == "chembl"


class TestAmbiguity:
    """Ambiguity is decided by structural disagreement, not by candidate count.

    Several records under one preferred name are normally duplicate deposits of
    the same molecule. Genuine ambiguity looks like kisspeptin, where ChEMBL
    holds a ~1302 Da and a ~5857 Da entity under the same name.
    """

    def _kisspeptin_like(self) -> StubChembl:
        return StubChembl(
            {
                "kisspeptin": [
                    mol("CHEMBL_K10", "KISSPEPTIN", max_phase=1.0, full_mwt=1302.46),
                    mol("CHEMBL_K54", "KISSPEPTIN", max_phase=0.0, full_mwt=5857.0),
                ]
            }
        )

    def test_structurally_divergent_same_name_is_ambiguous(self):
        ident = resolver(self._kisspeptin_like()).resolve("kisspeptin", "kisspeptin")
        assert ident.status is IdentityStatus.AMBIGUOUS
        # The higher recorded phase still wins the field values.
        assert ident.chembl_id.value == "CHEMBL_K10"

    def test_ambiguity_caveat_names_the_competing_records(self):
        ident = resolver(self._kisspeptin_like()).resolve("kisspeptin", "kisspeptin")
        caveat = " ".join(ident.caveats)
        assert "structurally different" in caveat
        assert "CHEMBL_K10" in caveat and "CHEMBL_K54" in caveat

    def test_ambiguous_identity_blocks_structure_models(self):
        ident = resolver(self._kisspeptin_like()).resolve("kisspeptin", "kisspeptin")
        assert ident.blocks_structure_models()

    def test_differing_inchikey_skeletons_are_ambiguous(self):
        r = resolver(
            StubChembl(
                {
                    "x": [
                        mol("C1", "X", inchikey="AAAAAAAAAAAAAA-BBBBBBBBBB-N"),
                        mol("C2", "X", inchikey="ZZZZZZZZZZZZZZ-BBBBBBBBBB-N"),
                    ]
                }
            )
        )
        assert r.resolve("x", "x").status is IdentityStatus.AMBIGUOUS

    def test_duplicate_deposits_resolve_rather_than_flagging_ambiguity(self):
        """Same name, consistent weight: a duplicate deposit, not an ambiguity."""
        r = resolver(
            StubChembl(
                {
                    "semaglutide": [
                        mol("CHEMBL_A", "SEMAGLUTIDE", max_phase=4.0, full_mwt=4113.6),
                        mol("CHEMBL_B", "SEMAGLUTIDE", max_phase=0.0, full_mwt=4113.6),
                    ]
                }
            )
        )
        ident = r.resolve("semaglutide", "semaglutide")
        assert ident.status is IdentityStatus.RESOLVED
        assert ident.chembl_id.value == "CHEMBL_A"
        assert any("duplicate deposits" in c for c in ident.caveats)

    def test_single_candidate_raises_no_duplicate_caveat(self):
        r = resolver(StubChembl({"x": [mol("C1", "X", max_phase=4.0)]}))
        ident = r.resolve("x", "x")
        assert not any("duplicate" in c for c in ident.caveats)
        assert ident.status is IdentityStatus.RESOLVED


class TestModalityAndCompleteness:
    def test_declared_modality_is_preserved(self):
        r = resolver(StubChembl({"x": [mol("C1", "X", molecule_type="Protein")]}))
        ident = r.resolve("x", "x", declared_modality=Modality.METAL_COMPLEX)
        assert ident.modality is Modality.METAL_COMPLEX

    def test_modality_inferred_from_chembl_when_undeclared(self):
        r = resolver(StubChembl({"x": [mol("C1", "X", molecule_type="Protein")]}))
        assert r.resolve("x", "x").modality is Modality.PROTEIN

    def test_peptidic_classification(self):
        assert Modality.MODIFIED_PEPTIDE.is_peptidic
        assert Modality.PROTEIN.is_peptidic
        assert not Modality.SMALL_MOLECULE.is_peptidic
        assert not Modality.METAL_COMPLEX.is_peptidic

    def test_partial_identity_reports_partial_completeness(self):
        r = resolver(StubChembl({"retatrutide": [mol("CHEMBL5095485", "RETATRUTIDE", max_phase=3.0)]}))
        ident = r.resolve("retatrutide", "retatrutide")
        assert 0.0 < ident.completeness < 1.0
        assert not ident.is_approved_drug
        assert ident.blocks_structure_models()

    def test_withdrawn_flag_surfaces_as_caveat(self):
        r = resolver(StubChembl({"x": [mol("C1", "X", max_phase=4.0, withdrawn=True)]}))
        assert any("withdrawn" in c for c in r.resolve("x", "x").caveats)
