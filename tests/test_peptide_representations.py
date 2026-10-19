"""Stage B representations: the properties the benchmark's conclusions rest on.

The benchmark's claims are only meaningful if the representations behave as
designed — positional rather than order-free, chemistry-aware rather than
twenty-letter, and generalising rather than memorising. Those are the
properties asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas.ml.peptide_representations import (
    C_TERM_SLOTS, FRAGMENT_BITS, N_TERM_SLOTS, build_representations,
    canonical_sequence, oriented_keys, positional_fragment_fingerprint,
    positional_onehot, residue_vocabulary, topology_features,
)
from atlas.ml.peptide_structure import analyse_peptide

from tests.test_peptide_structure import (
    AFAMELANOTIDE, ASPIRIN, BRADYKININ, CYCLOSPORINE, MELANOTAN_II, OXYTOCIN,
    SOMATOSTATIN,
)

ALA_GLY_SER = "C[C@H](N)C(=O)NCC(=O)N[C@@H](CO)C(O)=O"
SER_GLY_ALA = "OC[C@H](N)C(=O)NCC(=O)N[C@@H](C)C(O)=O"


@pytest.fixture(scope="module")
def peptides():
    return [analyse_peptide(s) for _c, s in (
        OXYTOCIN, SOMATOSTATIN, AFAMELANOTIDE, MELANOTAN_II, BRADYKININ,
        CYCLOSPORINE,
    )]


# ------------------------------------------------- the canonical collapse ----

def test_canonical_sequence_hides_non_canonical_residues():
    """What a twenty-letter alphabet destroys, stated as a test.

    Afamelanotide's norleucine and melanotan II's bridged residues have no
    canonical letter, so they become X.
    """
    afa = analyse_peptide(AFAMELANOTIDE[1])
    assert afa.sequence == "SYSXEHFRWGKPV"
    assert canonical_sequence(afa) == "SYSXEHFRWGKPV"
    assert "X" in canonical_sequence(analyse_peptide(MELANOTAN_II[1]))


def test_canonical_sequence_collapses_distinct_chemistry():
    """Two different side chains at one position give one canonical string.

    This is the mechanism behind the measured collision of 10 molecules
    spanning 4.01 log units of affinity.
    """
    chloro = "CC(N)C(=O)NC(Cc1ccc(Cl)cc1)C(=O)NCC(O)=O"
    naphthyl = "CC(N)C(=O)NC(Cc1ccc2ccccc2c1)C(=O)NCC(O)=O"
    a, b = (analyse_peptide(s) for s in (chloro, naphthyl))
    assert canonical_sequence(a) == canonical_sequence(b)
    assert tuple(oriented_keys(a)) != tuple(oriented_keys(b))


def test_fully_canonical_peptide_is_unchanged():
    somato = analyse_peptide(SOMATOSTATIN[1])
    assert canonical_sequence(somato) == somato.sequence == "AGCKNFFWKTFTSC"
    assert "X" not in canonical_sequence(somato)


# ------------------------------------------------------------- positional ----

def test_positional_encodings_distinguish_order():
    """The property Stage A's composition representation lacked."""
    a, b = (analyse_peptide(s) for s in (ALA_GLY_SER, SER_GLY_ALA))
    assert sorted(a.sequence) == sorted(b.sequence) and a.sequence != b.sequence
    vocab = residue_vocabulary([a, b])
    assert not np.allclose(*positional_onehot([a, b], vocab))
    assert not np.allclose(*positional_fragment_fingerprint([a, b]))


def test_cyclic_rotation_is_canonicalised():
    """A head-to-tail cyclic backbone has no first residue.

    Rotating to a canonical form is what makes a position index comparable
    across the series; without it the tracer's starting residue is an artefact
    of atom ordering.
    """
    cyc = analyse_peptide(CYCLOSPORINE[1])
    assert cyc.is_cyclic_backbone
    keys = oriented_keys(cyc)
    rotations = [tuple(keys[i:] + keys[:i]) for i in range(len(keys))]
    assert tuple(keys) == min(rotations)


def test_linear_peptide_keeps_its_real_n_terminus():
    """Rotation must not be applied where the N-terminus is real."""
    brad = analyse_peptide(BRADYKININ[1])
    assert not brad.is_cyclic_backbone
    assert oriented_keys(brad) == brad.residue_keys


def test_encoding_widths_are_as_declared(peptides):
    vocab = residue_vocabulary(peptides)
    slots = N_TERM_SLOTS + C_TERM_SLOTS
    assert positional_onehot(peptides, vocab).shape == (len(peptides), slots * len(vocab))
    assert positional_fragment_fingerprint(peptides).shape == (
        len(peptides), slots * FRAGMENT_BITS)


def test_short_peptides_are_padded_not_dropped():
    tri = analyse_peptide(ALA_GLY_SER)
    x = positional_fragment_fingerprint([tri])
    assert x.shape[0] == 1
    assert np.isfinite(x).all()


# ---------------------------------------------------------- generalisation ----

def test_onehot_is_blind_to_an_unseen_side_chain():
    """The mechanism by which one-hot must fail cross-series.

    A vocabulary fitted without the naphthyl side chain encodes a molecule
    carrying it as all-zero at that position.
    """
    chloro = analyse_peptide("CC(N)C(=O)NC(Cc1ccc(Cl)cc1)C(=O)NCC(O)=O")
    naphthyl = analyse_peptide("CC(N)C(=O)NC(Cc1ccc2ccccc2c1)C(=O)NCC(O)=O")
    vocab = residue_vocabulary([chloro])          # naphthyl deliberately unseen
    enc = positional_onehot([naphthyl], vocab)
    width = len(vocab)
    # Position 1 holds the unseen residue; its block carries no information.
    assert enc[0, width:2 * width].sum() == 0


def test_fragment_fingerprint_places_unseen_near_seen():
    """A fragment fingerprint generalises where a one-hot index cannot.

    4-chloro and 3-chloro phenylalanine share most bits; 4-chloro and glycine
    do not. This is why the fragment encoding wins cross-series.
        """
    p_cl = analyse_peptide("CC(N)C(=O)NC(Cc1ccc(Cl)cc1)C(=O)NCC(O)=O")
    m_cl = analyse_peptide("CC(N)C(=O)NC(Cc1cccc(Cl)c1)C(=O)NCC(O)=O")
    gly = analyse_peptide("CC(N)C(=O)NCC(=O)NCC(O)=O")
    x = positional_fragment_fingerprint([p_cl, m_cl, gly])

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    assert cos(x[0], x[1]) > cos(x[0], x[2])


# ------------------------------------------------------------------ bundle ----

def test_topology_features_are_finite_and_capture_constraint(peptides):
    t = topology_features(peptides)
    assert t.shape == (len(peptides), 12)
    assert np.isfinite(t).all()
    # Oxytocin and somatostatin carry disulfides; bradykinin does not.
    assert t[0, 3] == 1 and t[1, 3] == 1 and t[4, 3] == 0


def test_build_representations_aligns_every_matrix(peptides):
    reps = build_representations(peptides)
    assert reps.n_molecules == len(peptides)
    for name in reps.names:
        assert reps[name].shape[0] == len(peptides), name
        assert np.isfinite(reps[name]).all(), name


def test_build_representations_accepts_an_extra_block(peptides):
    extra = np.ones((len(peptides), 4))
    reps = build_representations(peptides, extra={"morgan_fp": extra})
    assert "morgan_fp" in reps.names
    assert "positional_fragment_fp+morgan_fp" in reps.names
    assert reps["positional_fragment_fp+morgan_fp"].shape[1] == (
        reps["positional_fragment_fp"].shape[1] + 4)


def test_vocabulary_from_training_only_excludes_test_residues(peptides):
    """Fitting the vocabulary on all molecules would let the test set define
    the feature space, which is leakage even with no label touched."""
    train, test = peptides[:3], peptides[3:]
    v_train = residue_vocabulary(train)
    v_all = residue_vocabulary(peptides)
    assert len(v_train) < len(v_all)
    test_only = {k for a in test for k in oriented_keys(a)} - set(v_train)
    assert test_only, "fixture no longer exercises unseen residues"


def test_non_peptide_yields_empty_keys():
    a = analyse_peptide(ASPIRIN[1])
    assert a.residue_keys == []
    assert canonical_sequence(a) == ""
    assert oriented_keys(a) == []
