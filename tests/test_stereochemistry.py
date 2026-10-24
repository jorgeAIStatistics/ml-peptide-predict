"""Backbone configuration: the D/L assignment and what it separates.

The assignment is deliberately not CIP-based, because L-cysteine is (R) where
the other L residues are (S). The reference battery below therefore includes
cysteine, and a CIP-based implementation would fail on it.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest
from rdkit import Chem

from atlas.ml.peptide_representations import (
    STEREO_FEATURE_NAMES, oriented_configs, stereo_features,
    stereo_oriented_keys, residue_vocabulary,
)
from atlas.ml.peptide_structure import analyse_peptide
from atlas.ml.stereochemistry import (
    Configuration, alpha_configurations, configuration_at,
    configuration_census, stereo_residue_keys,
)

# SMILES in PubChem's canonical orientation for each standard, so the
# reference is independent of how this module happens to write them.
_L_STANDARDS = {
    "L-alanine": "C[C@@H](C(=O)O)N",
    "L-cysteine": "C([C@@H](C(=O)O)N)S",      # CIP (R), configurationally L
    "L-serine": "C([C@@H](C(=O)O)N)O",
    "L-threonine": "C[C@H]([C@@H](C(=O)O)N)O",
    "L-proline": "C1C[C@H](NC1)C(=O)O",
    "L-tryptophan": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",
    "L-aspartate": "C([C@@H](C(=O)O)N)C(=O)O",
}
_D_STANDARDS = {
    "D-alanine": "C[C@H](C(=O)O)N",
    "D-cysteine": "C([C@H](C(=O)O)N)S",
    "D-phenylalanine": "C1=CC=C(C=C1)C[C@H](C(=O)O)N",
}

_ALPHA = Chem.MolFromSmarts("[NX3;!$([NX3]=*);!$([nX3])][CX4][CX3](=[OX1])")


def _alphas(smiles: str) -> dict[int, Configuration]:
    mol = Chem.MolFromSmiles(smiles)
    idx = {m[1] for m in mol.GetSubstructMatches(_ALPHA)}
    return alpha_configurations(mol, sorted(idx))


# ------------------------------------------------------ reference standards ----

@pytest.mark.parametrize("name,smiles", sorted(_L_STANDARDS.items()))
def test_l_standards_are_called_l(name, smiles):
    assert list(_alphas(smiles).values()) == [Configuration.L], name


@pytest.mark.parametrize("name,smiles", sorted(_D_STANDARDS.items()))
def test_d_standards_are_called_d(name, smiles):
    assert list(_alphas(smiles).values()) == [Configuration.D], name


def test_cysteine_is_the_case_a_cip_mapping_would_get_wrong():
    """L-cysteine is (R) at the alpha carbon while L-alanine is (S).

    Both must come back L. If this ever fails while the alanine test passes,
    the implementation has reverted to reading CIP labels.
    """
    cys = Chem.MolFromSmiles(_L_STANDARDS["L-cysteine"])
    Chem.AssignStereochemistry(cys, cleanIt=True, force=True)
    labels = {a.GetPropsAsDict().get("_CIPCode") for a in cys.GetAtoms()
              if a.HasProp("_CIPCode")}
    assert labels == {"R"}, "reference SMILES is not the (R) cysteine"
    assert list(_alphas(_L_STANDARDS["L-cysteine"]).values()) == [Configuration.L]
    assert list(_alphas(_L_STANDARDS["L-alanine"]).values()) == [Configuration.L]


def test_glycine_has_no_stereocentre_to_report():
    assert list(_alphas("C(C(=O)O)N").values()) == [Configuration.ACHIRAL]


def test_alpha_disubstituted_is_not_forced_into_the_binary():
    """Aib has no alpha hydrogen, so neither D nor L describes it."""
    assert list(_alphas("CC(C)(C(=O)O)N").values()) == [
        Configuration.ALPHA_DISUBSTITUTED]


def test_undefined_stereocentre_is_not_assumed_to_be_l():
    assert list(_alphas("CC(N)C(=O)O").values()) == [Configuration.UNSPECIFIED]
    assert not Configuration.UNSPECIFIED.is_defined
    assert Configuration.L.is_defined and Configuration.D.is_defined


def test_configuration_at_matches_the_batch_call():
    mol = Chem.MolFromSmiles(_L_STANDARDS["L-alanine"])
    idx = mol.GetSubstructMatches(_ALPHA)[0][1]
    assert configuration_at(mol, idx) is Configuration.L


def test_assignment_is_independent_of_atom_order():
    """The same molecule written two ways must give the same answer.

    This is the property the implementation buys by matching a chiral query
    rather than computing a permutation parity by hand: the depositor's atom
    ordering must not reach the result.
    """
    for smiles in list(_L_STANDARDS.values()) + list(_D_STANDARDS.values()):
        mol = Chem.MolFromSmiles(smiles)
        want = list(_alphas(smiles).values())
        for seed in range(5):
            shuffled = Chem.MolToSmiles(Chem.RenumberAtoms(
                mol, np.random.default_rng(seed).permutation(
                    mol.GetNumAtoms()).tolist()))
            assert list(_alphas(shuffled).values()) == want, smiles


# ------------------------------------------------------- traced into peptides ----

def test_a_dipeptide_reports_one_configuration_per_residue():
    a = analyse_peptide("C[C@@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)O)N")
    assert a.sequence == "AF"
    assert a.residue_configs == ["L", "L"]
    assert len(a.alpha_carbon_indices) == len(a.residue_keys)


def test_configurations_align_with_the_residue_keys():
    """Position 3 of one list must describe position 3 of the other."""
    a = analyse_peptide(
        "C[C@@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@H](C)C(=O)O)N")
    assert a.sequence == "AFA"
    assert a.residue_configs == ["L", "L", "D"]
    assert a.n_d_residues == 1


def test_ndp_alpha_msh_is_d_at_position_seven():
    """The substitution that defines the series this dataset is built from.

    NDP-alpha-MSH is Ac-Ser-Tyr-Ser-Nle-Glu-His-D-Phe-Arg-Trp-Gly-Lys-Pro-Val,
    and alpha-MSH is the same chain with L-Phe(7) and Met(4). Both are in the
    MC4R dataset; this test pins the one structural fact that distinguishes
    them. Nle traces as X, having no canonical side chain.
    """
    # Built residue by residue, so the test states the structure it means.
    smiles = (
        "CC(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CO)C(=O)"
        "N[C@@H](CCCC)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)"
        "N[C@H](Cc1ccccc1)C(=O)N[C@@H](CCCNC(N)=N)C(=O)"
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)NCC(=O)N[C@@H](CCCCN)C(=O)"
        "N1CCC[C@H]1C(=O)N[C@@H](C(C)C)C(N)=O"
    )
    ndp = analyse_peptide(smiles)
    assert ndp.sequence == "SYSXEHFRWGKPV"
    assert ndp.residue_configs[6] == Configuration.D.value
    assert ndp.n_d_residues == 1
    assert ndp.residue_configs[9] == Configuration.ACHIRAL.value   # Gly(10)


def test_unspecified_configuration_raises_a_caveat():
    a = analyse_peptide("CC(N)C(=O)NC(C)C(=O)NC(C)C(=O)O")
    assert a.n_unspecified_configurations == 3
    assert any("unspecified" in c for c in a.caveats)


# ------------------------------------------------------------ residue tokens ----

def test_d_and_l_residues_become_different_tokens():
    keys = stereo_residue_keys(["F", "F"], ["L", "D"])
    assert keys[0] != keys[1]
    assert keys[0] == "F", "L is the unmarked default"


def test_token_lengths_must_agree():
    """Zipping to the shorter list would attach configurations to the wrong
    positions, which is worse than failing."""
    with pytest.raises(ValueError):
        stereo_residue_keys(["F", "F"], ["L"])


def test_stereo_vocabulary_separates_the_two_phenylalanines():
    l_phe = analyse_peptide(
        "C[C@@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](C)C(=O)O)N")
    d_phe = analyse_peptide(
        "C[C@@H](C(=O)N[C@H](Cc1ccccc1)C(=O)N[C@@H](C)C(=O)O)N")
    assert l_phe.sequence == d_phe.sequence == "AFA"
    blind = residue_vocabulary([l_phe, d_phe])
    aware = residue_vocabulary([l_phe, d_phe], stereo=True)
    assert len(aware) > len(blind)
    assert stereo_oriented_keys(l_phe) != stereo_oriented_keys(d_phe)


def test_cyclic_rotation_is_shared_between_keys_and_configurations():
    """Slot 3 of one representation must be slot 3 of the others.

    The rotation that canonicalises a head-to-tail cyclic peptide is chosen
    once; choosing it separately per representation would misalign the stereo
    block against the fragment fingerprint whenever configuration broke a tie
    the constitution did not.
    """
    cyc = analyse_peptide(
        "C1[C@@H](C)C(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@H](C)C(=O)N1")
    keys = [k for k in stereo_oriented_keys(cyc)]
    cfgs = oriented_configs(cyc)
    assert len(keys) == len(cfgs) == len(cyc.residue_keys)
    for key, cfg in zip(keys, cfgs):
        if cfg == Configuration.D.value:
            assert key.startswith("D-")
        else:
            assert not key.startswith("D-")


# ---------------------------------------------------------- feature block ----

def test_stereo_block_width_matches_its_names():
    a = analyse_peptide("C[C@@H](C(=O)N[C@H](Cc1ccccc1)C(=O)O)N")
    x = stereo_features([a])
    assert x.shape == (1, len(STEREO_FEATURE_NAMES))


def test_all_l_peptide_has_only_the_all_l_flag_set():
    a = analyse_peptide("C[C@@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)O)N")
    x = stereo_features([a])[0]
    names = dict(zip(STEREO_FEATURE_NAMES, x))
    assert names["all_l"] == 1.0
    assert names["n_d_residues"] == 0.0
    assert names["d_fraction"] == 0.0


def test_a_d_residue_sets_its_slot_and_the_counts():
    a = analyse_peptide("C[C@@H](C(=O)N[C@H](Cc1ccccc1)C(=O)O)N")
    x = stereo_features([a])[0]
    names = dict(zip(STEREO_FEATURE_NAMES, x))
    assert names["n_d_residues"] == 1.0
    assert names["d_fraction"] == pytest.approx(0.5)
    assert names["all_l"] == 0.0
    assert names["slot1_is_D"] == 1.0


def test_counts_cover_residues_outside_the_indexed_slots():
    """A D residue at position 10 of a 13-mer falls in no slot; the
    molecule-level count must still see it."""
    from atlas.ml.peptide_representations import N_TERM_SLOTS, C_TERM_SLOTS
    # Nine L alanines, a D alanine, then three more L: the eight N-terminal
    # slots cover positions 1-8 and the three C-terminal ones 11-13, so
    # position 10 is indexed by neither.
    units = ["N[C@@H](C)C(=O)"] * 13
    units[9] = "N[C@H](C)C(=O)"
    a = analyse_peptide("".join(units) + "O")
    assert len(a.residue_keys) == 13
    assert a.residue_configs[9] == "D"
    x = stereo_features([a])[0]
    names = dict(zip(STEREO_FEATURE_NAMES, x))
    slots = [names[f"slot{i}_is_D"]
             for i in range(N_TERM_SLOTS + C_TERM_SLOTS)]
    assert sum(slots) == 0.0, "position 10 should fall outside every slot"
    assert names["n_d_residues"] == 1.0


def test_census_reports_fractions_and_disclaims_provenance():
    c = configuration_census([["L", "L", "D"], ["L", "achiral"]])
    assert c["n_molecules"] == 2
    assert c["n_alpha_carbons"] == 5
    assert c["molecules_with_a_d_residue"] == 1
    assert c["per_residue_counts"]["L"] == 3
    assert "not determined experimentally" in c["provenance"]
