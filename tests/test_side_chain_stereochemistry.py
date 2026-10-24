"""Side-chain stereochemistry: the census that decided not to fix it.

Two reference cases carry this file. The first is the loss itself: threonine
and allo-threonine, isoleucine and allo-isoleucine, where the deployed residue
key and the fragment fingerprint are identical for two different molecules.

The second is the case a naive fix gets wrong. D-threonine's beta carbon is
inverted relative to L-threonine's, so its *fragment* is the same as
allo-L-threonine's: an implementation that read the stereo-aware fragment as
"the residue's configuration" would call D-threonine and allo-L-threonine the
same residue. Only the fragment together with the alpha-carbon assignment
separates them, which is why these tests always assert both.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from atlas.ml.features import morgan_matrix
from atlas.ml.peptide_representations import (
    positional_fragment_fingerprint, stereo_oriented_keys,
)
from atlas.ml.peptide_structure import analyse_peptide, side_chain_atoms
from atlas.ml.side_chain_stereochemistry import (
    UNFRAGMENTED, side_chain_census, side_chain_collisions,
    side_chain_fragments, side_chain_stereoisomers,
)

# PubChem's canonical orientation for each standard, so the reference does not
# depend on how this project happens to write them.
STANDARDS = {
    "L-threonine": "C[C@H]([C@@H](C(=O)O)N)O",            # (2S,3R)
    "allo-L-threonine": "C[C@@H]([C@@H](C(=O)O)N)O",       # (2S,3S)
    "D-threonine": "C[C@@H]([C@H](C(=O)O)N)O",             # (2R,3S)
    "L-isoleucine": "CC[C@H](C)[C@@H](C(=O)O)N",           # (2S,3S)
    "allo-L-isoleucine": "CC[C@@H](C)[C@@H](C(=O)O)N",     # (2S,3R)
    "L-alanine": "C[C@@H](C(=O)O)N",
    "glycine": "C(C(=O)O)N",
}

#: Ala-Thr-Ala, all alpha carbons L, differing only in the threonine beta
#: centre. Built residue by residue so the test states the structure it means.
_ATA = "C[C@@H](C(=O)N[C@@H]([C{tag}H](C)O)C(=O)N[C@@H](C)C(=O)O)N"
ATA_THR = _ATA.format(tag="@@")
ATA_ALLO_THR = _ATA.format(tag="@")


def fragment(smiles: str, position: int = 0):
    """The side-chain fragment of one residue of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    analysis = analyse_peptide(smiles)
    if analysis.alpha_carbon_indices:
        ca = analysis.alpha_carbon_indices[position]
    else:                       # a free amino acid traces no chain
        ca = sorted(side_chain_atoms(mol))[position]
    return side_chain_fragments(mol)[ca]


# ------------------------------------------------------ reference standards ----

def test_the_cut_fragments_are_the_expected_side_chains():
    assert fragment(STANDARDS["L-alanine"]).flat_smiles == "*C"
    assert fragment(STANDARDS["glycine"]).is_empty
    assert fragment(STANDARDS["L-threonine"]).flat_smiles == "*C(C)O"
    assert fragment(STANDARDS["L-isoleucine"]).flat_smiles == "*C(C)CC"


@pytest.mark.parametrize("a,b", [
    ("L-threonine", "allo-L-threonine"),
    ("L-isoleucine", "allo-L-isoleucine"),
])
def test_diastereomers_share_a_flat_fragment_and_differ_with_stereochemistry(a, b):
    """The loss, and the instrument's ability to see it, in one assertion."""
    fa, fb = fragment(STANDARDS[a]), fragment(STANDARDS[b])
    assert fa.flat_smiles == fb.flat_smiles, "not a diastereomeric pair"
    assert fa.stereo_smiles != fb.stereo_smiles
    assert fa.carries_configuration and fb.carries_configuration


def test_d_threonine_and_allo_threonine_share_a_fragment():
    """The case a naive fix gets wrong.

    Inverting the alpha carbon inverts the beta carbon with it, so D-threonine
    and allo-L-threonine have the same side-chain fragment. A residue key built
    from the fragment alone would merge them; the fragment plus the
    alpha-carbon assignment does not.
    """
    d_thr, allo = fragment(STANDARDS["D-threonine"]), fragment(
        STANDARDS["allo-L-threonine"])
    assert d_thr.stereo_smiles == allo.stereo_smiles
    assert (analyse_peptide(STANDARDS["D-threonine"]).residue_configs
            != analyse_peptide(STANDARDS["allo-L-threonine"]).residue_configs)


def test_the_cut_does_not_depend_on_the_atom_order():
    """The property the cut buys over a rebuild.

    Rebuilding a fragment atom by atom reorders a stereocentre's neighbours, so
    a chiral tag copied across is right or wrong depending on the order the
    depositor happened to write. Cutting the bonds and letting RDKit assign the
    tags removes that dependence, and this test is what would catch a
    regression to a hand-rolled copy.
    """
    for name, smiles in STANDARDS.items():
        mol = Chem.MolFromSmiles(smiles)
        want = fragment(smiles).stereo_smiles
        for seed in range(5):
            order = np.random.default_rng(seed).permutation(
                mol.GetNumAtoms()).tolist()
            shuffled = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order))
            assert fragment(shuffled).stereo_smiles == want, f"{name}/{seed}"


# -------------------------------------------- the loss in the deployed rows ----

def test_the_deployed_representation_merges_the_two_threonines():
    """Both mechanisms, measured rather than described.

    The residue key is the flat fragment and the fingerprint generator runs
    with chirality off, so two molecules that differ at the threonine beta
    carbon receive the same key, the same configuration-aware key and the same
    bits -- in the fragment fingerprint and in the whole-molecule fingerprint
    alike.
    """
    thr, allo = analyse_peptide(ATA_THR), analyse_peptide(ATA_ALLO_THR)
    assert Chem.CanonSmiles(ATA_THR) != Chem.CanonSmiles(ATA_ALLO_THR)
    assert thr.residue_keys == allo.residue_keys == ["A", "T", "A"]
    assert thr.residue_configs == allo.residue_configs == ["L", "L", "L"]
    assert stereo_oriented_keys(thr) == stereo_oriented_keys(allo)

    rows = positional_fragment_fingerprint([thr, allo])
    assert np.array_equal(rows[0], rows[1]), "fragment fingerprint separates them"
    morgan = morgan_matrix([Chem.MolFromSmiles(ATA_THR),
                            Chem.MolFromSmiles(ATA_ALLO_THR)])
    assert np.array_equal(morgan[0], morgan[1]), "whole-molecule fp separates them"


def test_a_stereocentre_can_be_one_only_in_the_whole_molecule():
    """A side chain cut at both ends can lose a centre it really has.

    Residue 2's side chain bridges to residue 3's nitrogen, so cutting it out
    gives ``*CC(O)C*``: two branches that are identical inside the fragment and
    distinct outside it. The tag is dropped, correctly -- the fragment has no
    stereocentre there -- which means no fragment-local residue key can carry
    this configuration, configuration-aware or not. 41 of the MC4R set's
    side-chain stereocentres are of this kind, and the whole-molecule
    stereoisomer census is what covers them.
    """
    bridged = "C[C@@H](N)C(=O)N[C@@H](C[C@H](O)C1)C(=O)N1[C@@H](C)C(=O)O"
    f = fragment(bridged, position=1)
    assert f.flat_smiles == "*CC(O)C*"
    assert f.n_specified_centres == 1
    assert not f.carries_configuration


def test_the_instrument_is_finer_than_the_deployed_key():
    """Cutting every boundary bond separates what the deployed key merges.

    Proline's side chain closes onto its own nitrogen and gets a reserved
    letter from the tracer; the cut reports the fragment with a dummy at each
    end. That the instrument's partition is finer is what makes a null result
    on it the stronger statement.
    """
    pro = "C[C@@H](C(=O)N1CCC[C@H]1C(=O)O)N"
    analysis = analyse_peptide(pro)
    assert analysis.residue_keys[1] == "P"
    assert fragment(pro, position=1).flat_smiles.count("*") == 2


# ----------------------------------------------------------------- census ----

def test_the_census_finds_the_ambiguous_key():
    """Two molecules, one key: the census must name the pair it cannot split."""
    thr, allo = analyse_peptide(ATA_THR), analyse_peptide(ATA_ALLO_THR)
    c = side_chain_census([thr, allo], np.array([8.0, 6.0]), ["thr", "allo"])
    assert c["n_residues_traced"] == 6
    assert c["residues_with_a_side_chain_stereocentre"] == 2
    assert c["flat_fragments_seen_in_more_than_one_configuration"] == 1
    assert len(c["ambiguous_keys"]) == 1
    key = c["ambiguous_keys"][0]
    assert key["alpha_configuration"] == "L"
    assert key["flat_fragment"] == "*C(C)O"
    assert key["n_residues"] == 2
    assert key["n_residues_in_the_minority_configuration"] == 1
    # Which of the two configurations is "the minority" is a tie here; what
    # the census must get right is the residue it points at.
    assert len(key["minority_carriers"]) == 1
    carrier = key["minority_carriers"][0]
    assert carrier["molecule"] in ("thr", "allo")
    assert (carrier["position"], carrier["residue_key"]) == (2, "T")


def test_the_collision_detector_fires_when_a_collision_exists():
    """The dataset result below is only worth reading if this passes.

    A detector that returned zero unconditionally would agree with the MC4R
    census. These two molecules differ in nothing but the threonine beta
    centre, so one group must be reported, with the label spread it costs.
    """
    thr, allo = analyse_peptide(ATA_THR), analyse_peptide(ATA_ALLO_THR)
    y = np.array([8.0, 6.0])
    col = side_chain_collisions([thr, allo], y)
    assert col["groups_merged_by_side_chain_blindness"] == 1
    assert col["molecules_affected"] == 2
    assert col["max_pki_range_within_those_groups"] == pytest.approx(2.0)

    iso = side_chain_stereoisomers([thr, allo], y, ["thr", "allo"])
    assert iso["stereoisomeric_pairs"] == 1
    assert iso["pairs_separated_by_the_alpha_carbon_key"] == 0
    assert iso["pairs_needing_side_chain_configuration"] == 1
    assert iso["max_abs_delta_pki"] == pytest.approx(2.0)


def test_a_duplicated_molecule_is_not_a_stereoisomeric_pair():
    thr = analyse_peptide(ATA_THR)
    iso = side_chain_stereoisomers([thr, analyse_peptide(ATA_THR)],
                                   np.array([8.0, 8.0]))
    assert iso["stereoisomeric_pairs"] == 0


def test_an_unparseable_analysis_does_not_break_the_census():
    broken = analyse_peptide("not a molecule")
    c = side_chain_census([broken, analyse_peptide(ATA_THR)],
                          np.array([5.0, 8.0]))
    assert c["n_molecules"] == 2
    assert c["residues_with_a_side_chain_stereocentre"] == 1


def test_unfragmented_is_reported_rather_than_mistaken_for_glycine():
    assert UNFRAGMENTED.flat_smiles == "?"
    assert not UNFRAGMENTED.is_empty
    assert not UNFRAGMENTED.carries_configuration


# --------------------------------------------------------- the MC4R census ----

@pytest.fixture(scope="module")
def mc4r_census() -> dict:
    from atlas.ml.side_chain_stereochemistry import _load

    analyses, y, ids = _load()
    if len(analyses) < 100:
        pytest.skip("MC4R dataset not built")
    return side_chain_census(analyses, y, ids)


def test_the_set_does_carry_side_chain_stereocentres(mc4r_census):
    """The loss is not vacuous: the measurement had something to find."""
    assert mc4r_census["residues_with_a_side_chain_stereocentre"] > 100
    assert mc4r_census["molecules_with_a_side_chain_stereocentre"] > 50
    assert mc4r_census["residues_the_cut_could_not_isolate"] == 0


def test_only_the_threonine_key_is_ambiguous(mc4r_census):
    """The measured result, pinned.

    One (alpha configuration, flat fragment) pair in the whole set occurs in
    two side-chain configurations: L-threonine's. Every other side-chain
    stereocentre, the 160 noncanonical fragments included, occurs in one
    configuration only, so the flat key loses nothing that is actually varied.
    If this fails the census needs redoing -- see
    docs/ATLAS_2.5_STEREOCHEMISTRY.md section 7.
    """
    keys = mc4r_census["ambiguous_keys"]
    assert len(keys) == 1, [k["flat_fragment"] for k in keys]
    assert keys[0]["flat_fragment"] == "*C(C)O"
    assert keys[0]["alpha_configuration"] == "L"
    assert keys[0]["n_residues_in_the_minority_configuration"] == 7
    carriers = keys[0]["minority_carriers"]
    assert {c["residue_key"] for c in carriers} == {"T"}
    assert len({c["molecule"] for c in carriers}) == 7


def test_every_stereoisomeric_pair_is_separated_by_the_alpha_carbon_key(mc4r_census):
    """The decision this work rests on.

    25 constitutional groups hold 33 stereoisomeric pairs, and the
    alpha-carbon assignment separates all 33. Not one pair in the set needs
    side-chain configuration, so making the fragment key configuration-aware
    would change no row and could not change a metric. A failure here means a
    pair now exists and the fix has to be reconsidered.
    """
    iso = mc4r_census["stereoisomers"]
    assert iso["stereoisomeric_pairs"] == 33
    assert iso["constitutional_groups_with_more_than_one_configuration"] == 25
    assert iso["pairs_needing_side_chain_configuration"] == 0, iso["pairs"]
    assert (iso["pairs_separated_by_the_alpha_carbon_key"]
            == iso["stereoisomeric_pairs"])


def test_no_group_is_merged_by_side_chain_blindness(mc4r_census):
    """The same conclusion on the finer, fragment-level partition."""
    col = mc4r_census["collisions"]
    assert col["groups_merged_by_side_chain_blindness"] == 0
    assert col["molecules_affected"] == 0
    assert col["max_pki_range_within_those_groups"] == 0.0


def test_the_census_does_not_claim_the_configurations_were_measured(mc4r_census):
    assert "not determined experimentally" in mc4r_census["provenance"]
