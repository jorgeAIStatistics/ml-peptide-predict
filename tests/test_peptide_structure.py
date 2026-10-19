"""Validation of the peptide backbone tracer.

Every structure below was retrieved from PubChem by CID and is recorded here
verbatim so the suite runs offline. The expected residue counts and sequences
are independently known properties of these molecules, not outputs of this
code: oxytocin is CYIQNCPLG, somatostatin-14 is AGCKNFFWKTFTSC, angiotensin II
is DRVYIHPF, bradykinin is RPPGFSPFR. A test that compared the tracer against
its own output would establish nothing.
"""

from __future__ import annotations

import pytest

from atlas.ml.peptide_structure import (
    CANONICAL_SIDE_CHAINS, PeptideClass, analyse_peptide, _tautomer_key,
)

# (cid, name, smiles) as retrieved from PubChem PUG REST.
GLYGLY = (11163, "C(C(=O)NCC(=O)O)N")
GHK = (73587, "C1=C(NC=N1)C[C@@H](C(=O)N[C@@H](CCCCN)C(=O)O)NC(=O)CN")
GLUTATHIONE = (124886, "C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N")
OXYTOCIN = (439302, "CC[C@H](C)[C@H]1C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CSSC[C@@H](C(=O)N[C@H](C(=O)N1)CC2=CC=C(C=C2)O)N)C(=O)N3CCC[C@H]3C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N)CC(=O)N)CCC(=O)N")
LEUPROLIDE = (657181, "CCNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCN=C(N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](CC2=CC=C(C=C2)O)NC(=O)[C@H](CO)NC(=O)[C@H](CC3=CNC4=CC=CC=C43)NC(=O)[C@H](CC5=CN=CN5)NC(=O)[C@@H]6CCC(=O)N6")
BREMELANOTIDE = (9941379, "CCCC[C@@H](C(=O)N[C@H]1CC(=O)NCCCC[C@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](NC(=O)[C@@H](NC1=O)CC2=CN=CN2)CC3=CC=CC=C3)CCCN=C(N)N)CC4=CNC5=CC=CC=C54)C(=O)O)NC(=O)C")
MELANOTAN_II = (92432, "CCCC[C@@H](C(=O)N[C@H]1CC(=O)NCCCC[C@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@H](NC(=O)[C@@H](NC1=O)CC2=CN=CN2)CC3=CC=CC=C3)CCCN=C(N)N)CC4=CNC5=CC=CC=C54)C(=O)N)NC(=O)C")
AFAMELANOTIDE = (16197727, "CCCC[C@@H](C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N[C@H](CC2=CC=CC=C2)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CC3=CNC4=CC=CC=C43)C(=O)NCC(=O)N[C@@H](CCCCN)C(=O)N5CCC[C@H]5C(=O)N[C@@H](C(C)C)C(=O)N)NC(=O)[C@H](CO)NC(=O)[C@H](CC6=CC=C(C=C6)O)NC(=O)[C@H](CO)NC(=O)C")
ASPIRIN = (2244, "CC(=O)OC1=CC=CC=C1C(=O)O")
SILDENAFIL = (135398744, "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C")
CYCLOSPORINE = (5284373, "CC[C@H]1C(=O)N(CC(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N1)[C@@H]([C@H](C)C/C=C/C)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C")
SOMATOSTATIN = (16129706, "C[C@H]([C@H]1C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](CSSC[C@@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N1)CCCCN)CC2=CNC3=CC=CC=C32)CC4=CC=CC=C4)CC5=CC=CC=C5)CC(=O)N)CCCCN)NC(=O)CNC(=O)[C@H](C)N)C(=O)O)CO)[C@@H](C)O)CC6=CC=CC=C6)O")
ANGIOTENSIN_II = (172198, "CC[C@H](C)[C@@H](C(=O)N[C@@H](CC1=CN=CN1)C(=O)N2CCC[C@H]2C(=O)N[C@@H](CC3=CC=CC=C3)C(=O)O)NC(=O)[C@H](CC4=CC=C(C=C4)O)NC(=O)[C@H](C(C)C)NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@H](CC(=O)O)N")
BRADYKININ = (439201, "C1C[C@H](N(C1)C(=O)[C@@H]2CCCN2C(=O)[C@H](CCCN=C(N)N)N)C(=O)NCC(=O)N[C@@H](CC3=CC=CC=C3)C(=O)N[C@@H](CO)C(=O)N4CCC[C@H]4C(=O)N[C@@H](CC5=CC=CC=C5)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O")


@pytest.mark.parametrize("cid_smiles,expected", [
    (GLYGLY, 2), (GHK, 3), (GLUTATHIONE, 2), (OXYTOCIN, 9), (LEUPROLIDE, 9),
    (BREMELANOTIDE, 7), (MELANOTAN_II, 7), (AFAMELANOTIDE, 13),
    (CYCLOSPORINE, 11), (SOMATOSTATIN, 14), (ANGIOTENSIN_II, 8),
    (BRADYKININ, 9), (ASPIRIN, 0), (SILDENAFIL, 0),
])
def test_residue_count(cid_smiles, expected):
    """Backbone tracing recovers the known residue count."""
    _cid, smiles = cid_smiles
    assert analyse_peptide(smiles).n_residues_in_longest_chain == expected


@pytest.mark.parametrize("cid_smiles,sequence", [
    (OXYTOCIN, "CYIQNCPLG"),
    (SOMATOSTATIN, "AGCKNFFWKTFTSC"),
    (ANGIOTENSIN_II, "DRVYIHPF"),
    (BRADYKININ, "RPPGFSPFR"),
    (GHK, "GHK"),
    (GLYGLY, "GG"),
])
def test_sequence_recovered(cid_smiles, sequence):
    """Side-chain identification recovers the known one-letter sequence."""
    _cid, smiles = cid_smiles
    assert analyse_peptide(smiles).sequence == sequence


def test_leucine_is_not_isoleucine():
    """Oxytocin position 8 is leucine, and must not be read as isoleucine.

    Capped with a methyl rather than a dummy atom, both side chains become
    2-methylbutane and the attachment point that distinguishes them is lost.
    """
    seq = analyse_peptide(OXYTOCIN[1]).sequence
    assert seq[7] == "L"
    assert _tautomer_key("*CC(C)C") != _tautomer_key("*C(C)CC")


def test_arginine_matches_either_guanidine_tautomer():
    """Arginine is written with the double bond on either nitrogen.

    Both records below are real: afamelanotide's deposits CCCNC(=N)N and
    leuprolide's deposits CCCN=C(N)N. Tautomer normalisation is what makes the
    same residue match in both.
    """
    assert "R" in (analyse_peptide(LEUPROLIDE[1]).sequence or "")
    assert "R" in (analyse_peptide(AFAMELANOTIDE[1]).sequence or "")
    assert _tautomer_key("*CCCNC(=N)N") == _tautomer_key("*CCCN=C(N)N")


def test_afamelanotide_is_norleucine_substituted():
    """Afamelanotide is [Nle4, D-Phe7]-alpha-MSH: position 4 is not canonical."""
    a = analyse_peptide(AFAMELANOTIDE[1])
    assert a.sequence == "SYSXEHFRWGKPV"
    assert "*CCCC" in a.unmatched_fragments  # n-butyl: norleucine


def test_glutathione_has_two_alpha_backbone_residues():
    """Glutathione is a tripeptide, but only two residues are alpha-linked.

    Its glutamate is joined through the gamma carboxyl, so strict alpha
    backbone tracing finds a two-residue chain inside three amide units. The
    discrepancy is reported as a caveat rather than silently resolved.
    """
    a = analyse_peptide(GLUTATHIONE[1])
    assert a.n_residues_in_longest_chain == 2
    assert a.n_backbone_units == 3
    assert any("longest traceable chain" in c for c in a.caveats)


@pytest.mark.parametrize("cid_smiles", [ASPIRIN, SILDENAFIL])
def test_non_peptides_are_not_peptides(cid_smiles):
    a = analyse_peptide(cid_smiles[1])
    assert a.peptide_class is PeptideClass.SMALL_MOLECULE
    assert not a.is_peptide


def test_disulfide_detected():
    for smiles, n in ((OXYTOCIN[1], 1), (SOMATOSTATIN[1], 1)):
        a = analyse_peptide(smiles)
        assert a.n_disulfides == n
        assert a.peptide_class is PeptideClass.DISULFIDE_CONSTRAINED


def test_cyclosporine_is_head_to_tail_cyclic():
    a = analyse_peptide(CYCLOSPORINE[1])
    assert a.is_cyclic_backbone
    assert a.peptide_class is PeptideClass.CYCLIC


def test_lactam_bridge_detected_not_counted_as_head_to_tail():
    """Melanotan II closes through an Asp-Lys side-chain lactam, not head to tail."""
    a = analyse_peptide(MELANOTAN_II[1])
    assert not a.is_cyclic_backbone
    assert a.n_side_chain_bridges == 2  # the two residues the bridge joins
    assert a.peptide_class is PeptideClass.CYCLIC


def test_side_chain_walk_does_not_cross_disulfide():
    """Both oxytocin cysteines are identified despite the bridge joining them.

    Without cutting the walk at the S-S bond, a cysteine side chain walks
    across into the far half of the molecule and matches nothing.
    """
    assert analyse_peptide(OXYTOCIN[1]).sequence.count("C") == 2


def test_unparseable_structure_abstains():
    a = analyse_peptide("not a structure")
    assert a.peptide_class is PeptideClass.UNPARSEABLE
    assert not a.is_peptide
    assert a.caveats


def test_empty_smiles_abstains():
    assert analyse_peptide("").peptide_class is PeptideClass.UNPARSEABLE


def test_all_canonical_side_chains_are_distinct():
    """No two canonical residues may share a comparison key."""
    keys = {}
    for code, smi in CANONICAL_SIDE_CHAINS.items():
        key = _tautomer_key("*" + smi)
        assert key not in keys, f"{code} collides with {keys.get(key)}"
        keys[key] = code
