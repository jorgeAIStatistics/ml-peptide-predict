"""The structurally derived binding pocket: counting, mapping and transfer.

Coordinate files are large and gitignored, so these tests work from the saved
pocket definition plus unit tests of the pure functions. Two bugs found while
building it are pinned: a Counter fed a dict summed distances instead of
counting structures, and the reference receptor's identical-residue count was
never set.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.receptor_pocket import (
    CONTACT_CUTOFF_A, CUTOFF_SWEEP, MC4R_PEPTIDE_COMPLEXES, MIN_STRUCTURES,
    POCKET_PATH, PocketDefinition, load_pocket, pocket_encoding,
    pocket_residues_for_receptor,
)


@pytest.fixture(scope="module")
def pocket() -> PocketDefinition:
    if not POCKET_PATH.exists():
        pytest.skip(f"{POCKET_PATH.name} not built; run atlas.ml.receptor_pocket")
    return load_pocket()


# ------------------------------------------------------- support counting ----

def test_support_cannot_exceed_the_number_of_structures(pocket):
    """The bug this pins: Counter.update(dict) adds values, not keys.

    Fed {position: distance} it summed distances, producing 'support' of 41
    from ten structures.
    """
    n_structures = len(pocket.per_structure)
    assert n_structures > 0
    for position, support in pocket.support.items():
        assert 1 <= support <= n_structures, (position, support)


def test_every_consensus_position_meets_the_support_threshold(pocket):
    assert pocket.min_structures == MIN_STRUCTURES
    for position in pocket.positions:
        assert pocket.support[str(position)] >= pocket.min_structures


def test_support_is_an_integer_count(pocket):
    for support in pocket.support.values():
        assert isinstance(support, int) or float(support).is_integer()


def test_positions_are_sorted_and_unique(pocket):
    assert pocket.positions == sorted(set(pocket.positions))


# --------------------------------------------------------------- geometry ----

def test_cutoff_is_recorded_and_sweep_is_monotone(pocket):
    """A larger distance cutoff cannot find fewer contacts."""
    assert pocket.cutoff_angstrom == CONTACT_CUTOFF_A
    sweep = {float(k): v for k, v in pocket.cutoff_sensitivity.items()}
    assert set(sweep) == set(CUTOFF_SWEEP)
    sizes = [sweep[c] for c in sorted(sweep)]
    assert sizes == sorted(sizes), sweep


def test_pocket_size_at_the_chosen_cutoff_matches_the_sweep(pocket):
    assert pocket.cutoff_sensitivity[str(pocket.cutoff_angstrom)] == len(pocket.positions)


def test_every_structure_contributed_contacts(pocket):
    for s in pocket.per_structure:
        assert s["contacts"], s["pdb_id"]
        assert all(0.0 < float(d) <= pocket.cutoff_angstrom
                   for d in s["contacts"].values()), s["pdb_id"]


def test_peptide_chains_are_full_length_not_truncated(pocket):
    """Filtering the peptide chain by standard-amino-acid membership silently
    dropped two of setmelanotide's eight residues. Every structure's peptide
    must have at least three residues, and the known lengths must appear."""
    lengths = {s["pdb_id"]: s["n_peptide_residues"] for s in pocket.per_structure}
    assert all(n >= 3 for n in lengths.values()), lengths
    if "7F55" in lengths:
        assert lengths["7F55"] == 7        # bremelanotide
    if "7PIU" in lengths:
        assert lengths["7PIU"] == 8        # setmelanotide
    if "7F53" in lengths:
        assert lengths["7F53"] >= 13       # alpha-MSH


def test_fusion_constructs_map_only_partially(pocket):
    """The X-ray entries carry a glycogen-synthase partner.

    Author numbering could not be trusted because of exactly this, so the
    partial mapping is the expected, correct behaviour.
    """
    by_id = {s["pdb_id"]: s for s in pocket.per_structure}
    for pdb_id in ("6W25", "8WKY", "8WKZ"):
        if pdb_id not in by_id:
            continue
        s = by_id[pdb_id]
        assert s["n_receptor_residues_mapped"] < s["n_receptor_residues_observed"]
        assert s["caveats"]


def test_engineered_mutations_are_counted_not_hidden(pocket):
    """Identity comes from the reference sequence, geometry from the structure."""
    by_id = {s["pdb_id"]: s for s in pocket.per_structure}
    xray = [by_id[p] for p in ("6W25", "8WKY", "8WKZ") if p in by_id]
    if xray:
        assert any(s["letter_mismatches"] > 0 for s in xray)
    cryo = [s for s in pocket.per_structure
            if s.get("method") == "cryo-EM"]
    if cryo:
        assert all(s["letter_mismatches"] == 0 for s in cryo), (
            "cryo-EM entries should match the reference sequence exactly")


# --------------------------------------------------------------- transfer ----

def test_reference_receptor_is_identical_to_itself(pocket):
    """The second bug pinned: this reported 0 of 31 identical."""
    ref = pocket.transferred[pocket.reference_receptor]
    assert ref["identity_to_reference"] == 1.0
    assert ref["n_gaps"] == 0
    assert ref["n_identical_pocket_residues"] == len(pocket.positions)


def test_transfer_covers_every_consensus_position(pocket):
    for name, t in pocket.transferred.items():
        if t.get("residues") is None:
            continue
        assert len(t["residues"]) == len(pocket.positions), name


def test_unmappable_position_is_a_gap_not_a_guess(pocket):
    """An absent counterpart must be recorded, never invented."""
    for name, t in pocket.transferred.items():
        if t.get("residues") is None:
            continue
        assert t["n_gaps"] == sum(1 for r in t["residues"] if r == "-"), name


def test_receptors_differ_at_pocket_positions(pocket):
    """If every receptor had the same pocket, the encoder would carry nothing."""
    strings = {
        n: "".join(t["residues"]) for n, t in pocket.transferred.items()
        if t.get("residues")
    }
    assert len(set(strings.values())) == len(strings)


def test_pocket_identity_is_lower_than_perfect_for_other_receptors(pocket):
    for name, t in pocket.transferred.items():
        if name == pocket.reference_receptor or not t.get("residues"):
            continue
        assert t["n_identical_pocket_residues"] < len(pocket.positions), name


# --------------------------------------------------------------- encoding ----

def test_pocket_encoding_width_and_finiteness(pocket):
    from atlas.ml.residue_chemistry import DESCRIPTOR_NAMES

    names = ["MC4R", "MC3R", "MC4R"]
    enc = pocket_encoding(names, pocket)
    assert enc.shape == (3, len(pocket.positions) * len(DESCRIPTOR_NAMES))
    assert np.isfinite(enc).all()
    # The same receptor must encode identically.
    assert np.allclose(enc[0], enc[2])


def test_pocket_encoding_separates_receptors(pocket):
    names = [n for n, t in pocket.transferred.items() if t.get("residues")]
    enc = pocket_encoding(names, pocket)
    rows = {tuple(np.round(r, 6)) for r in enc}
    assert len(rows) == len(names)


def test_unknown_receptor_encodes_as_zeros(pocket):
    enc = pocket_encoding(["not-a-receptor"], pocket)
    assert enc.shape[0] == 1
    assert enc.sum() == 0.0


def test_pocket_residues_accessor(pocket):
    assert pocket_residues_for_receptor(pocket, "MC4R")
    assert pocket_residues_for_receptor(pocket, "nonexistent") is None


# ----------------------------------------------------------- provenance ----

def test_ten_complexes_are_declared_with_method_and_resolution():
    assert len(MC4R_PEPTIDE_COMPLEXES) == 10
    for pdb_id, meta in MC4R_PEPTIDE_COMPLEXES.items():
        assert len(pdb_id) == 4
        assert meta["method"] in ("cryo-EM", "X-ray")
        assert 2.0 < meta["resolution"] < 4.0
        assert meta["ligand"]


def test_duplicate_ligands_are_visible(pocket):
    """Ten entries are not ten independent observations."""
    ligands = [s.get("ligand") for s in pocket.per_structure if s.get("ligand")]
    assert len(set(ligands)) < len(ligands)


def test_caveats_record_what_a_contact_is_not(pocket):
    joined = " ".join(pocket.caveats).lower()
    assert "energetic contribution" in joined
    assert "alignment" in joined


def test_reference_accession_is_recorded(pocket):
    assert pocket.reference_accession == "P32245"


def test_saved_definition_round_trips(pocket, tmp_path):
    path = pocket.save(tmp_path / "pocket.json")
    again = load_pocket(path)
    assert again.positions == pocket.positions
    assert again.support == pocket.support
    assert json.loads(path.read_text())["reference_receptor"] == "MC4R"
