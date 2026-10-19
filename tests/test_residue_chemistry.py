"""Stage 2 residue chemistry engine: provenance, ionisation, and coverage.

The engine's reason for existing is that it describes non-canonical residues,
which a tabulated amino-acid scale cannot. These tests assert that, assert the
provenance discipline the brief requires, and pin the known divergences so a
computed descriptor is never later mistaken for a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas.ml.peptide_structure import analyse_peptide
from atlas.ml.residue_chemistry import (
    ASSUMED_PH, DESCRIPTOR_NAMES, REFERENCE_PKA, Provenance, describe_residue,
    ionised_fraction, positional_chemistry, provenance_summary,
    residue_property_matrix,
)

from tests.test_peptide_structure import (
    AFAMELANOTIDE, ASPIRIN, MELANOTAN_II, OXYTOCIN, SOMATOSTATIN,
)

P_CHLORO_PHE = "*Cc1ccc(Cl)cc1"
DICHLORO_PHE = "*Cc1ccc(Cl)cc1Cl"
NAPHTHYL_ALA = "*Cc1ccc2ccccc2c1"
NORLEUCINE = "*CCCC"


# -------------------------------------------------------------- provenance ----

def test_nothing_is_labelled_measured():
    """No descriptor here is an experimental observation.

    Calling a Crippen logP contribution a measured hydrophobicity is exactly
    the conflation the platform exists to prevent.
    """
    for key in ("A", "W", "K", P_CHLORO_PHE, NAPHTHYL_ALA):
        p = describe_residue(key)
        assert Provenance.MEASURED not in p.provenance.values(), key


def test_ionisation_is_estimated_not_computed():
    """Charge depends on an assumed pH and reference pKa, so it is an estimate."""
    p = describe_residue("K")
    for field in ("estimated_charge", "is_acidic", "is_basic", "is_ionisable"):
        assert p.provenance[field] is Provenance.ESTIMATED
    # Whereas a graph property is computed.
    assert p.provenance["heavy_atoms"] is Provenance.COMPUTED


def test_glycine_descriptors_are_unavailable_not_zero():
    """Glycine has no side chain, which is different from having a null one."""
    p = describe_residue("G")
    assert p.glycine_like == 1.0
    assert not p.has_side_chain
    assert p.provenance["crippen_logp"] is Provenance.UNAVAILABLE
    assert p.provenance["side_chain_volume"] is Provenance.UNAVAILABLE
    assert p.caveats


def test_proline_records_its_backbone_constraint():
    p = describe_residue("P")
    assert p.backbone_constrained == 1.0
    assert p.provenance["crippen_logp"] is Provenance.UNAVAILABLE
    assert any("backbone nitrogen" in c for c in p.caveats)


def test_unavailable_zeros_remain_distinguishable_in_the_vector():
    """A model sees the vector, not the provenance dict.

    The two flags must therefore be inside the vector, or glycine's zeros are
    indistinguishable from a real zero.
    """
    assert "glycine_like" in DESCRIPTOR_NAMES
    assert "backbone_constrained" in DESCRIPTOR_NAMES
    g = describe_residue("G").vector()
    p = describe_residue("P").vector()
    assert g[DESCRIPTOR_NAMES.index("glycine_like")] == 1.0
    assert p[DESCRIPTOR_NAMES.index("backbone_constrained")] == 1.0
    assert not np.allclose(g, p)


def test_unbuildable_side_chain_abstains():
    p = describe_residue("*not-a-fragment")
    assert all(v is Provenance.UNAVAILABLE for v in p.provenance.values())
    assert p.caveats


# --------------------------------------------------------------- ionisation ----

def test_henderson_hasselbalch_is_half_at_the_pka():
    """The defining property of the relation, both directions."""
    assert ionised_fraction(7.0, 7.0, acidic=True) == pytest.approx(0.5)
    assert ionised_fraction(7.0, 7.0, acidic=False) == pytest.approx(0.5)


def test_acid_and_base_respond_to_ph_in_opposite_directions():
    acid = [ionised_fraction(6.0, ph, acidic=True) for ph in (4, 6, 8)]
    base = [ionised_fraction(6.0, ph, acidic=False) for ph in (4, 6, 8)]
    assert acid == sorted(acid)            # acid ionises more as pH rises
    assert base == sorted(base, reverse=True)


def test_charged_residues_carry_the_expected_sign_at_physiological_ph():
    for acidic in ("D", "E"):
        assert describe_residue(acidic).estimated_charge < -0.9
        assert describe_residue(acidic).is_acidic == 1.0
    for basic in ("K", "R"):
        assert describe_residue(basic).estimated_charge > 0.9
        assert describe_residue(basic).is_basic == 1.0
    for neutral in ("A", "L", "F", "S"):
        assert describe_residue(neutral).estimated_charge == pytest.approx(0.0, abs=0.05)


def test_cysteine_is_only_slightly_ionised_at_physiological_ph():
    """A thiol at pKa 8.3 is about 11% ionised at pH 7.4, not 50%.

    An earlier three-bucket rule counted anything within one pH unit of its
    pKa as half charged and put cysteine at -0.5.
    """
    c = describe_residue("C").estimated_charge
    assert -0.2 < c < -0.05
    assert c == pytest.approx(
        -ionised_fraction(REFERENCE_PKA["thiol"], ASSUMED_PH, acidic=True), abs=1e-6)


def test_histidine_is_ph_sensitive_and_mostly_neutral_at_7_4():
    """The microenvironment case the brief singles out."""
    charges = [describe_residue("H", ph).estimated_charge
               for ph in (5.0, 6.0, 7.0, 7.4, 8.0)]
    assert charges == sorted(charges, reverse=True)       # monotone in pH
    assert describe_residue("H", REFERENCE_PKA["imidazole"]).estimated_charge == \
        pytest.approx(0.5, abs=0.01)
    assert describe_residue("H", 7.4).estimated_charge < 0.1


def test_partial_ionisation_is_flagged_as_a_caveat():
    assert any("sensitive to the" in c for c in describe_residue("H").caveats)
    assert not describe_residue("K").caveats     # fully ionised, nothing to warn


# ----------------------------------------------- non-canonical coverage ----

def test_non_canonical_residues_are_described_not_refused():
    """The engine's purpose: a lookup table has no row for these."""
    for key in (P_CHLORO_PHE, DICHLORO_PHE, NAPHTHYL_ALA, NORLEUCINE):
        p = describe_residue(key)
        assert not p.is_canonical
        assert p.provenance["crippen_logp"] is Provenance.COMPUTED
        assert p.heavy_atoms > 0
        assert any("non-canonical" in c for c in p.caveats)


def test_halogen_substitution_is_counted_and_shifts_lipophilicity():
    """A computed descriptor responds to substitution; a tabulated one cannot.

    These are the two commonest non-canonical residues in the dataset, 237 and
    228 occurrences, and they differ by one chlorine.
    """
    phe = describe_residue("F")
    mono = describe_residue(P_CHLORO_PHE)
    di = describe_residue(DICHLORO_PHE)
    assert (phe.halogens, mono.halogens, di.halogens) == (0.0, 1.0, 2.0)
    assert phe.crippen_logp < mono.crippen_logp < di.crippen_logp


def test_naphthylalanine_has_two_aromatic_rings():
    assert describe_residue(NAPHTHYL_ALA).aromatic_rings == 2.0
    assert describe_residue("F").aromatic_rings == 1.0


def test_norleucine_differs_from_leucine_despite_equal_formula():
    """Both are C4H9 side chains; only their branching differs."""
    nle, leu = describe_residue(NORLEUCINE), describe_residue("L")
    assert nle.heavy_atoms == leu.heavy_atoms
    assert nle.branching_index < leu.branching_index


# ----------------------------------------------- hydrophobicity behaviour ----

def test_hydrophobic_residues_rank_above_charged_ones():
    """A coarse ordering robust enough to assert.

    Fine-grained orderings are deliberately not asserted: see the divergence
    test below.
    """
    hydrophobic = [describe_residue(r).crippen_logp for r in ("W", "F", "I", "L", "V")]
    polar = [describe_residue(r).crippen_logp for r in ("S", "T", "N", "Q", "R")]
    assert min(hydrophobic) > max(polar)


def test_computed_logp_is_not_an_experimental_hydrophobicity_scale():
    """A known divergence, pinned so it is never mistaken for agreement.

    Fragment Crippen logP ranks tyrosine above leucine, because the phenol ring
    contributes substantial calculated lipophilicity. Experimental transfer
    scales generally rank leucine higher. The engine hard-codes no published
    scale and does not claim to reproduce one.
    """
    assert describe_residue("Y").crippen_logp > describe_residue("L").crippen_logp


def test_arginine_is_less_lipophilic_than_lysine():
    assert describe_residue("R").crippen_logp < describe_residue("K").crippen_logp


# ----------------------------------------------------------- aggregation ----

def test_residue_property_matrix_shape_and_finiteness():
    keys = ["A", "W", "G", "P", P_CHLORO_PHE]
    m = residue_property_matrix(keys)
    assert m.shape == (len(keys), len(DESCRIPTOR_NAMES))
    assert np.isfinite(m).all()


def test_empty_key_list_returns_empty_matrix():
    assert residue_property_matrix([]).shape == (0, len(DESCRIPTOR_NAMES))


def test_positional_chemistry_is_aligned_and_finite():
    peptides = [analyse_peptide(s) for _c, s in
                (OXYTOCIN, SOMATOSTATIN, AFAMELANOTIDE, MELANOTAN_II)]
    x = positional_chemistry(peptides)
    assert x.shape[0] == len(peptides)
    assert np.isfinite(x).all()
    assert x.shape[1] == (8 + 3) * len(DESCRIPTOR_NAMES) + 2 * len(DESCRIPTOR_NAMES) + 1


def test_positional_chemistry_distinguishes_residue_order():
    """Positional, as Stage A's measurement requires."""
    a = analyse_peptide("C[C@H](N)C(=O)NCC(=O)N[C@@H](CO)C(O)=O")   # Ala-Gly-Ser
    b = analyse_peptide("OC[C@H](N)C(=O)NCC(=O)N[C@@H](C)C(O)=O")   # Ser-Gly-Ala
    assert sorted(a.sequence) == sorted(b.sequence)
    assert not np.allclose(*positional_chemistry([a, b]))


def test_non_peptide_gives_a_zero_row_not_an_error():
    x = positional_chemistry([analyse_peptide(ASPIRIN[1])])
    assert x.shape[0] == 1
    assert np.isfinite(x).all()
    assert x.sum() == 0.0


def test_provenance_summary_reports_coverage():
    peptides = [analyse_peptide(s) for _c, s in
                (OXYTOCIN, AFAMELANOTIDE, MELANOTAN_II)]
    s = provenance_summary(peptides)
    assert s["n_residue_positions"] > 0
    assert s["noncanonical_positions"] > 0
    assert 0.0 < s["noncanonical_fraction"] < 1.0
    assert s["assumed_ph"] == ASSUMED_PH
    assert s["reference_pka"] == REFERENCE_PKA


def test_describe_residue_is_deterministic():
    assert describe_residue(DICHLORO_PHE).vector().tolist() == \
        describe_residue(DICHLORO_PHE).vector().tolist()
