"""Typed interaction descriptors: chemistry, thresholds and the leakage guard.

Coordinate files are gitignored, so these tests work from the saved profiles
plus unit tests of the pure geometric functions on synthetic coordinates.

The leakage guard is the most important thing here. These descriptors are
computed from experimentally determined bound complexes; letting one reach a
prospective prediction pathway would make that evaluation meaningless, and the
brief forbids it explicitly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.interaction_features import (
    CRITERIA, INTERACTION_PATH, InferenceMode, InteractionProfile,
    ProspectiveLeakError, _count_salt_bridges, _find_rings, _infer_bonds,
    _interplanar_angle, _planar_ring, _ring_frame, _type_peptide,
    assert_retrospective_only, load_profiles,
)


@pytest.fixture(scope="module")
def report() -> dict:
    if not INTERACTION_PATH.exists():
        pytest.skip(f"{INTERACTION_PATH.name} not built; run atlas.ml.interaction_features")
    return load_profiles()


# ------------------------------------------------------- the leakage guard ----

def test_prospective_mode_is_refused():
    with pytest.raises(ProspectiveLeakError, match="bound complex"):
        assert_retrospective_only(InferenceMode.PROSPECTIVE)
    with pytest.raises(ProspectiveLeakError):
        assert_retrospective_only("prospective")


def test_retrospective_mode_is_permitted():
    assert assert_retrospective_only(InferenceMode.RETROSPECTIVE_STRUCTURE) is None
    assert assert_retrospective_only("retrospective_structure") is None


def test_unknown_mode_is_refused_not_ignored():
    with pytest.raises(ValueError):
        assert_retrospective_only("whatever")


def test_feature_vector_refuses_a_prospective_profile():
    """A silent fallback is what would let a bound structure into a de novo
    prediction, so the vector itself checks."""
    p = InteractionProfile(
        pdb_id="TEST", ligand="x", method="x", resolution=2.0,
        receptor_chain="R", peptide_chain="P",
        mode=InferenceMode.PROSPECTIVE.value)
    with pytest.raises(ProspectiveLeakError):
        p.vector()


def test_every_saved_profile_is_marked_retrospective(report):
    assert report["mode"] == InferenceMode.RETROSPECTIVE_STRUCTURE.value
    assert "must not enter a prospective prediction pathway" in report["restriction"]
    for p in report["profiles"]:
        assert p["mode"] == InferenceMode.RETROSPECTIVE_STRUCTURE.value


# ------------------------------------------------------------- bond typing ----

def test_bond_inference_links_only_close_atoms():
    coords = np.array([[0.0, 0, 0], [1.5, 0, 0], [5.0, 0, 0]])
    neigh = _infer_bonds(coords, ["C", "C", "C"])
    assert neigh[0] == {1} and neigh[1] == {0}
    assert neigh[2] == set()


def test_bond_inference_handles_an_empty_set():
    assert _infer_bonds(np.zeros((0, 3)), []) == []


def test_guanidinium_nitrogens_are_not_also_primary_amines():
    """Regression: arginine was yielding three cationic groups, not one.

    NH1 and NH2 each have a single heavy neighbour, so they matched the
    primary-amine rule as well as the guanidinium rule, and one Arg-Asp salt
    bridge was counted three times.
    """
    class _Atom:
        def __init__(self, name, element, coord):
            self._name, self.element, self.coord = name, element, np.array(coord)

        def get_id(self):
            return self._name

    class _Res:
        id = (" ", 1, " ")

        def __init__(self, atoms):
            self._atoms = atoms

        def __iter__(self):
            return iter(self._atoms)

        def get_resname(self):
            return "ARG"

    # A guanidinium: CZ bonded to NE, NH1, NH2 at ~1.33 A.
    atoms = [
        _Atom("CZ", "C", [0.0, 0.0, 0.0]),
        _Atom("NE", "N", [1.33, 0.0, 0.0]),
        _Atom("NH1", "N", [-0.66, 1.15, 0.0]),
        _Atom("NH2", "N", [-0.66, -1.15, 0.0]),
    ]
    typed = _type_peptide([_Res(atoms)])
    assert len(typed.cationic) == 1, typed.cationic
    assert len(typed.cationic[0]) == 4          # the carbon and its three N


def test_carboxylate_is_detected_as_anionic():
    class _Atom:
        def __init__(self, name, element, coord):
            self._name, self.element, self.coord = name, element, np.array(coord)

        def get_id(self):
            return self._name

    class _Res:
        id = (" ", 1, " ")

        def __init__(self, atoms):
            self._atoms = atoms

        def __iter__(self):
            return iter(self._atoms)

        def get_resname(self):
            return "LIG"

    atoms = [
        _Atom("C", "C", [0.0, 0.0, 0.0]),
        _Atom("O1", "O", [1.25, 0.0, 0.0]),
        _Atom("O2", "O", [-0.63, 1.08, 0.0]),
    ]
    typed = _type_peptide([_Res(atoms)])
    assert len(typed.anionic) == 1
    assert len(typed.anionic[0]) == 3


# ------------------------------------------------------------ ring geometry ----

def test_planar_hexagon_is_accepted_and_a_puckered_one_rejected():
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    flat = np.stack([np.cos(angles) * 1.4, np.sin(angles) * 1.4, np.zeros(6)], axis=1)
    assert _planar_ring(flat, range(6))
    puckered = flat.copy()
    puckered[::2, 2] = 1.2
    assert not _planar_ring(puckered, range(6))


def test_ring_search_finds_a_six_membered_cycle():
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    coords = np.stack([np.cos(angles) * 1.4, np.sin(angles) * 1.4, np.zeros(6)], axis=1)
    neigh = _infer_bonds(coords, ["C"] * 6)
    rings = _find_rings(neigh, coords)
    assert len(rings) == 1
    assert sorted(rings[0]) == list(range(6))


def test_ring_frame_normal_is_perpendicular():
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    coords = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)], axis=1)
    centroid, normal = _ring_frame(coords, range(6))
    assert np.allclose(centroid, [0, 0, 0], atol=1e-6)
    assert abs(abs(normal[2]) - 1.0) < 1e-6


def test_interplanar_angle_extremes():
    z, x = np.array([0.0, 0, 1]), np.array([1.0, 0, 0])
    assert _interplanar_angle(z, z) == pytest.approx(0.0, abs=1e-6)
    assert _interplanar_angle(z, x) == pytest.approx(90.0, abs=1e-6)
    assert _interplanar_angle(z, -z) == pytest.approx(0.0, abs=1e-6)   # sign-free


# -------------------------------------------------------------- thresholds ----

def test_criteria_are_recorded_with_the_output(report):
    for key in ("hbond_donor_acceptor_max_a", "salt_bridge_max_a",
                "hydrophobic_max_a", "aromatic_centroid_max_a",
                "covalent_bond_max_a", "sasa_algorithm"):
        assert key in report["criteria"]
    assert report["criteria"] == CRITERIA


def test_hydrogen_bonds_are_labelled_candidates_because_there_are_no_hydrogens(report):
    assert "not applied" in str(CRITERIA["hbond_angular_criterion"])
    for p in report["profiles"]:
        assert "hbond_candidates" in p
        assert any("no hydrogens" in c for c in p["caveats"])


def test_counts_are_monotone_in_the_cutoff(report):
    for p in report["profiles"]:
        for kind, counts in p["cutoff_sensitivity"].items():
            ordered = [counts[k] for k in sorted(counts, key=float)]
            assert ordered == sorted(ordered), (p["pdb_id"], kind)


def test_the_setmelanotide_salt_bridge_is_a_threshold_effect(report):
    """21LT reports no salt bridge at 4.0 A and two at 4.5 A.

    Its nearest charged pair sits at 4.24 A where 7PIU's is at 2.54 A -- the
    same ligand on either side of the cutoff. This is why the sweep is saved.
    """
    by_id = {p["pdb_id"]: p for p in report["profiles"]}
    if "21LT" not in by_id:
        pytest.skip("21LT not profiled")
    p = by_id["21LT"]
    assert p["cutoff_sensitivity"]["salt_bridges"]["4.0"] == 0
    assert p["cutoff_sensitivity"]["salt_bridges"]["4.5"] >= 1
    assert 4.0 < p["nearest_distances"]["charged_pair_a"] < 4.5


def test_salt_bridge_counting_is_cutoff_monotone_as_a_unit():
    from atlas.ml.interaction_features import _AtomSet

    rec = _AtomSet(np.array([[0.0, 0, 0]]), ["O"], ["D1:OD1"], anionic=[[0]])
    pep = _AtomSet(np.array([[4.2, 0, 0]]), ["N"], ["R1:NZ"], cationic=[[0]])
    assert _count_salt_bridges(rec, pep, 4.0) == 0
    assert _count_salt_bridges(rec, pep, 4.5) == 1


# ------------------------------------------------------- measured findings ----

def test_solvent_accessibility_fractions_are_fractions(report):
    """Dividing the whole interface by one side's surface gave values above 1."""
    for p in report["profiles"]:
        frac = p["sasa"]["peptide_fraction_buried"]
        assert 0.0 <= frac <= 1.0, (p["pdb_id"], frac)


def test_buried_area_is_consistent_with_its_components(report):
    for p in report["profiles"]:
        s = p["sasa"]
        assert s["buried_peptide_a2"] == pytest.approx(
            s["sasa_peptide_alone_a2"] - s["sasa_peptide_in_complex_a2"], rel=1e-6)
        assert s["buried_surface_area_a2"] == pytest.approx(
            s["buried_peptide_a2"] + s["buried_receptor_a2"], rel=1e-6)
        assert s["interface_area_a2"] == pytest.approx(
            s["buried_surface_area_a2"] / 2.0, rel=1e-6)


def test_compact_cyclic_peptides_bury_more_of_themselves_than_the_13mers(report):
    """A chemically coherent check, not an assumption: a 13-mer keeps more
    surface solvent-exposed than a constrained heptapeptide."""
    by_len = {}
    for p in report["profiles"]:
        by_len.setdefault(p["n_peptide_residues"], []).append(
            p["sasa"]["peptide_fraction_buried"])
    short = [f for n, fs in by_len.items() if n <= 9 for f in fs]
    long_ = [f for n, fs in by_len.items() if n >= 13 for f in fs]
    if short and long_:
        assert min(short) > max(long_)


def test_cation_pi_absence_is_evidenced_by_distance(report):
    """Absence recorded with the measurement that establishes it.

    An expectation that Arg-aromatic contacts would be common was wrong: the
    nearest cationic group to receptor ring centroid is 6.8-8.6 A in most
    structures, well beyond any cation-pi range.
    """
    far = [p for p in report["profiles"]
           if (p["nearest_distances"]["cation_to_ring_centroid_a"] or 99) > 6.0]
    assert far, "no structure records a cation-ring distance"
    for p in far:
        assert p["cation_pi"] == 0, p["pdb_id"]


def test_every_interface_has_hydrophobic_contacts_and_buried_area(report):
    for p in report["profiles"]:
        assert p["hydrophobic_contacts"] > 0, p["pdb_id"]
        assert p["sasa"]["buried_surface_area_a2"] > 100, p["pdb_id"]


def test_peptide_rings_are_found_despite_non_canonical_residues(report):
    """Residue names carry no information for these ligands, so rings must come
    from the graph or they are invisible."""
    for p in report["profiles"]:
        assert p["interface_geometry"]["peptide_rings_found"] >= 3, p["pdb_id"]


def test_all_ten_structures_profiled_with_duplicate_ligands_visible(report):
    assert report["n_structures"] == 10
    assert report["n_distinct_ligands"] < report["n_structures"]
