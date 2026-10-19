"""The structural interface explorer: its payloads and its vocabulary.

Two kinds of test here, and the second is the point.

The first kind checks that the composed payloads carry what the explorer
needs: the ten profiles joined to the consensus pocket, residue chemistry per
position, the transfer to the other receptors, and geometry that can be drawn.

The second kind pins the distinctions the brief forbids collapsing. A
hydrogen bond at this resolution is a *candidate*, because no hydrogens are
present and no D-H...A angle was checked. A peptide-side charged group or ring
is *inferred* from geometry, because these ligands are largely non-canonical.
A pocket position on MC1R, MC3R or MC5R is *transferred* by alignment, because
no peptide-bound structure of those receptors exists. And none of it is an
affinity estimate: the peptide model fails its gate and publishes no number.
Those are the claims a pretty picture is most likely to quietly erase, so each
is asserted rather than left to the renderer.

Coordinate files are gitignored. Everything that needs one skips on a cold
cache rather than failing, exactly as the explorer degrades.
"""

from __future__ import annotations

import math

import pytest

from atlas.api.app import create_app
from atlas.api.interface_payload import (
    RENDERING_NOTE, STRUCTURE_EVIDENCE, interface_detail, interface_index,
)
from atlas.ml.interaction_features import InferenceMode, ProspectiveLeakError
from atlas.ml.interface_geometry import (
    HBOND_CANDIDATE_CAVEAT, INFERRED_CHEMISTRY_CAVEAT, INTERACTION_KINDS,
    InterfaceGeometryUnavailable, available_structures, geometry_for_structure,
)


@pytest.fixture(scope="module")
def index() -> dict:
    payload = interface_index()
    if not payload.get("available"):
        pytest.skip(payload.get("reason", "interface payloads not built"))
    return payload


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _a_cached_structure() -> str:
    for pdb_id, cached in available_structures().items():
        if cached:
            return pdb_id
    pytest.skip("no coordinate files cached; run atlas.ml.receptor_pocket")


# ------------------------------------------------------------ the index ----

class TestIndex:
    def test_joins_every_profile_to_its_pocket_contacts(self, index):
        assert index["n_structures"] == len(index["structures"])
        for s in index["structures"]:
            assert s["pdb_id"]
            # Every structure contributes contacts to the pocket it defines.
            assert s["n_pocket_contacts"] > 0
            assert s["n_pocket_contacts"] == len(s["contacts"])

    def test_pocket_positions_carry_support_and_transfer(self, index):
        positions = index["pocket"]["positions"]
        assert len(positions) == index["pocket"]["n_positions"]
        for p in positions:
            assert p["support"] >= index["pocket"]["min_structures"]
            assert p["residue"]
            # The reference receptor is not listed among the transfers.
            assert index["pocket"]["reference_receptor"] not in p["transferred"]

    def test_every_pocket_residue_has_computed_chemistry(self, index):
        by_residue = index["chemistry"]["by_residue"]
        for p in index["pocket"]["positions"]:
            assert p["residue"] in by_residue, p["position"]
        names = index["chemistry"]["descriptor_names"]
        for entry in by_residue.values():
            assert set(entry["descriptors"]) == set(names)

    def test_interaction_kind_order_survives_serialisation(self, client):
        """Flask sorts object keys, so the declared order is sent separately.

        The explorer presents the kinds in a deliberate order; without this
        list it would render them alphabetically, which puts "aromatic
        oblique" first and the salt bridges last.
        """
        payload = client.get("/api/interfaces").get_json()
        assert payload["interaction_kind_order"] == list(INTERACTION_KINDS)
        assert set(payload["interaction_kinds"]) == set(INTERACTION_KINDS)

    def test_coordinate_availability_is_reported_per_structure(self, index):
        cached = available_structures()
        for s in index["structures"]:
            assert s["coordinates_cached"] == cached[s["pdb_id"]]


# ------------------------------------------- the distinctions that matter ----

class TestVocabulary:
    def test_hydrogen_bonds_are_candidates_everywhere_they_appear(self, index):
        kind = index["interaction_kinds"]["hbond_candidate"]
        assert kind["evidence"] == "candidate"
        assert "hydrogen" in kind["caveat"].lower()
        assert "angle" in kind["caveat"].lower()
        # The saved profile's own criteria say the same thing.
        assert "not applied" in index["criteria"]["hbond_angular_criterion"]
        # And the field it reads is the candidate count, not a bond count.
        assert kind["profile_field"] == "hbond_candidates"

    def test_no_kind_is_labelled_experimental(self, index):
        """The structures are experimental; the typing laid over them is not.

        Calling a geometric contact "experimental" would be the exact
        conflation this explorer exists to avoid, so no interaction kind may
        carry that class even though every coordinate behind it is measured.
        """
        for key, spec in index["interaction_kinds"].items():
            assert spec["evidence"] in ("candidate", "geometric"), key

    def test_peptide_side_chemistry_is_marked_inferred(self, index):
        for key in ("salt_bridge", "cation_pi", "aromatic_stacked"):
            caveat = index["interaction_kinds"][key]["caveat"]
            assert caveat == INFERRED_CHEMISTRY_CAVEAT, key

    def test_transferred_positions_are_not_presented_as_observed(self, index):
        detail = STRUCTURE_EVIDENCE["transferred"]["detail"]
        assert "no experimental structure" in detail.lower()
        for name in ("MC1R", "MC3R", "MC5R"):
            assert name in detail
        # Every caveat on the pocket itself repeats the restriction.
        joined = " ".join(index["pocket"]["caveats"]).lower()
        assert "alignment" in joined

    def test_a_contact_is_never_called_an_energetic_contribution(self, index):
        joined = " ".join(index["pocket"]["caveats"]).lower()
        assert "not a measurement" in joined
        assert "energetic contribution" in joined

    def test_the_viewer_does_not_claim_to_be_a_graphics_package(self, index):
        note = index["rendering_note"]
        assert note == RENDERING_NOTE
        assert "not a molecular-graphics package" in note
        assert "no surface" in note

    def test_no_numeric_affinity_is_offered(self, index):
        mp = index["model_position"]
        assert mp["publishes_numeric_affinity"] is False
        assert mp["result"]["passes"] is False
        assert "no numeric affinity estimate" in mp["statement"]
        # The gate's failing criteria are carried, not just the verdict.
        criteria = mp["result"]["criteria"]
        assert criteria["cross_series_mae"]["met"] is False
        assert criteria["cross_series_r2"]["met"] is False

    def test_retrospective_restriction_travels_with_the_payload(self, index):
        assert index["mode"] == InferenceMode.RETROSPECTIVE_STRUCTURE.value
        assert "must not enter a prospective" in index["restriction"]
        assert "cannot be computed for an unsolved pair" in (
            index["model_position"]["retrospective_only"])

    def test_residue_chemistry_is_never_labelled_measured(self, index):
        provenance = set(index["chemistry"]["descriptor_provenance"].values())
        assert "measured" not in provenance
        assert provenance <= {"computed", "estimated"}

    def test_ionisation_descriptors_declare_their_assumed_ph(self, index):
        chem = index["chemistry"]
        for name in ("estimated_charge", "is_acidic", "is_basic", "is_ionisable"):
            assert chem["descriptor_provenance"][name] == "estimated"
        assert str(chem["assumed_ph"]) in chem["note"]


# ---------------------------------------------------------- the geometry ----

class TestGeometry:
    def test_derived_counts_match_the_published_profile(self, index):
        """The drawing and the saved profile must be the same result.

        The geometry module re-enumerates the pairs behind each saved count
        using the same criteria. If its total ever differs from the published
        one, the picture has quietly replaced the result, so the comparison is
        returned with every payload and asserted here across every cached
        structure.
        """
        checked = 0
        for s in index["structures"]:
            if not s["coordinates_cached"]:
                continue
            detail = interface_detail(s["pdb_id"])
            geometry = detail.get("geometry")
            assert geometry is not None, s["pdb_id"]
            for kind, agreement in geometry["agreement"].items():
                assert agreement["agrees"], (s["pdb_id"], kind, agreement)
                assert agreement["derived"] == agreement["published"]
            assert geometry["agrees_with_profile"] is True
            checked += 1
        if not checked:
            pytest.skip("no coordinate files cached")

    def test_every_pair_is_drawable_and_carries_its_position(self, index):
        pdb_id = _a_cached_structure()
        geometry = interface_detail(pdb_id)["geometry"]
        pocket = {p["position"] for p in index["pocket"]["positions"]}
        seen_in_pocket = 0
        for kind, pairs in geometry["interactions"].items():
            for pair in pairs:
                assert len(pair["a"]) == 3 and len(pair["b"]) == 3
                assert pair["distance_a"] > 0
                if pair["receptor_position"] in pocket:
                    seen_in_pocket += 1
        # Linking a residue to its contacts is the whole interaction; if no
        # pair resolved to a pocket position the link is broken.
        assert seen_in_pocket > 0

    def test_contact_distances_respect_their_stated_cutoffs(self, index):
        pdb_id = _a_cached_structure()
        geometry = interface_detail(pdb_id)["geometry"]
        limits = {
            "hbond_candidate": 3.5, "salt_bridge": 4.0, "hydrophobic": 4.5,
            "cation_pi": 6.0,
        }
        for kind, limit in limits.items():
            for pair in geometry["interactions"][kind]:
                assert pair["distance_a"] <= limit + 1e-6, (kind, pair)

    def test_aromatic_classes_follow_the_stated_angles(self, index):
        """Stacked below 30 degrees, T-shaped above 60, oblique between."""
        checked = 0
        for s in index["structures"]:
            if not s["coordinates_cached"]:
                continue
            geometry = interface_detail(s["pdb_id"])["geometry"]
            for pair in geometry["interactions"]["aromatic_stacked"]:
                assert pair["interplanar_deg"] <= 30.0
                checked += 1
            for pair in geometry["interactions"]["aromatic_tshaped"]:
                assert pair["interplanar_deg"] >= 60.0
                checked += 1
            for pair in geometry["interactions"]["aromatic_oblique"]:
                assert 30.0 < pair["interplanar_deg"] < 60.0
                checked += 1
        if not checked:
            pytest.skip("no aromatic pairs in the cached structures")

    def test_coordinates_are_centred_and_rounded(self, index):
        pdb_id = _a_cached_structure()
        geometry = interface_detail(pdb_id)["geometry"]
        atoms = geometry["peptide"]["atoms"]
        assert atoms
        for atom in atoms:
            for value in atom["xyz"]:
                assert abs(value) <= geometry["extent_a"] + 1e-6
                assert round(value, 2) == value
        # The translation is reported, so the deposited frame is recoverable.
        assert len(geometry["centroid"]) == 3

    def test_unaligned_residues_are_not_named_from_the_reference(self, index):
        """The X-ray constructs carry a fusion partner.

        Those residues align to nothing in MC4R, so they get a null position.
        Naming them from the reference sequence would invent an identity.
        """
        for s in index["structures"]:
            if not s["coordinates_cached"]:
                continue
            if s["n_receptor_residues_mapped"] == s["n_receptor_residues_observed"]:
                continue
            geometry = interface_detail(s["pdb_id"])["geometry"]
            unmapped = [t for t in geometry["receptor_trace"] if t["position"] is None]
            assert unmapped, s["pdb_id"]
            return
        pytest.skip("no fusion-construct entry cached")

    def test_peptide_residue_spans_cover_every_atom_once(self, index):
        pdb_id = _a_cached_structure()
        peptide = interface_detail(pdb_id)["geometry"]["peptide"]
        cursor = 0
        for res in peptide["residues"]:
            assert res["atom_from"] == cursor
            cursor = res["atom_to"]
        assert cursor == len(peptide["atoms"])

    def test_non_canonical_peptide_residues_are_flagged(self, index):
        """These ligands are largely non-canonical; that is why the chemistry
        is inferred rather than looked up, so the flag must survive."""
        for s in index["structures"]:
            if not s["coordinates_cached"]:
                continue
            peptide = interface_detail(s["pdb_id"])["geometry"]["peptide"]
            if any(not r["is_canonical"] for r in peptide["residues"]):
                return
        pytest.skip("no cached structure carries a non-canonical residue")

    def test_bonds_are_within_the_stated_covalent_cutoff(self, index):
        pdb_id = _a_cached_structure()
        peptide = interface_detail(pdb_id)["geometry"]["peptide"]
        xyz = [a["xyz"] for a in peptide["atoms"]]
        for i, j in peptide["bonds"]:
            assert math.dist(xyz[i], xyz[j]) <= 1.95 + 1e-6

    def test_a_missing_coordinate_file_is_an_absent_input(self, tmp_path):
        """Not an error, and never filled in with something else."""
        with pytest.raises(InterfaceGeometryUnavailable, match="fetch_structure"):
            geometry_for_structure("7PIU", "MVNSTHRGMH" * 3,
                                   path=tmp_path / "absent.cif")

    def test_geometry_refuses_a_prospective_mode(self, monkeypatch):
        """The guard that keeps bound-structure features out of prediction.

        It runs before the file is read, so it fires even on a cold cache.
        """
        import atlas.ml.interface_geometry as mod

        monkeypatch.setattr(
            mod, "assert_retrospective_only",
            lambda mode: (_ for _ in ()).throw(ProspectiveLeakError("guarded")))
        with pytest.raises(ProspectiveLeakError):
            mod.geometry_for_structure("7PIU", "MVNSTHRGMH")


# -------------------------------------------------------------- the API ----

class TestEndpoints:
    def test_index_endpoint_serves_the_joined_payload(self, client):
        res = client.get("/api/interfaces")
        assert res.status_code == 200
        payload = res.get_json()
        assert payload["available"] is True
        assert payload["structures"]
        assert payload["pocket"]["positions"]

    def test_detail_endpoint_is_case_insensitive(self, client, index):
        pdb_id = index["structures"][0]["pdb_id"]
        upper = client.get(f"/api/interfaces/{pdb_id}")
        lower = client.get(f"/api/interfaces/{pdb_id.lower()}")
        assert upper.status_code == lower.status_code == 200
        assert upper.get_json()["structure"]["pdb_id"] == pdb_id

    def test_unknown_structure_is_refused_not_invented(self, client):
        res = client.get("/api/interfaces/XXXX")
        assert res.status_code == 404
        assert res.get_json()["available"] is False

    def test_detail_carries_the_gate_so_a_picture_is_not_a_prediction(
            self, client, index):
        payload = client.get(
            f"/api/interfaces/{index['structures'][0]['pdb_id']}").get_json()
        assert payload["model_position"]["publishes_numeric_affinity"] is False
        assert payload["rendering_note"] == RENDERING_NOTE

    def test_a_cold_cache_degrades_rather_than_failing(self, client, monkeypatch):
        """A fresh checkout has no coordinate files. The structure page must
        still serve its saved profile and contact distances, and say plainly
        why the drawing is absent."""
        import atlas.api.interface_payload as payload_mod

        monkeypatch.setattr(
            payload_mod, "interface_detail", payload_mod.interface_detail)
        import atlas.ml.interface_geometry as geom

        monkeypatch.setattr(
            geom, "geometry_for_structure",
            lambda *a, **k: (_ for _ in ()).throw(
                InterfaceGeometryUnavailable("no coordinate file; fetch_structure")))
        payload_mod.interface_detail.cache_clear()
        try:
            payload = payload_mod.interface_detail("7PIU")
            assert payload["available"] is True
            assert payload["geometry"] is None
            assert "fetch_structure" in payload["geometry_unavailable"]
            # The saved result is unaffected by the absent file.
            assert payload["structure"]["hbond_candidates"] >= 0
            assert payload["structure"]["contacts"]
        finally:
            payload_mod.interface_detail.cache_clear()


# ------------------------------------------------- the frontend contract ----

class TestFrontendContract:
    """The explorer is served as ES modules with no build step, so a missing
    file or a duplicate element id is only caught by loading the page. These
    check what can be checked without a browser."""

    def test_the_workspace_modules_are_served(self, client):
        for name in ("js/interfaces.js", "js/structure3d.js"):
            res = client.get(f"/{name}")
            assert res.status_code == 200, name
            assert b"export" in res.data

    def test_the_shell_registers_the_workspace(self, client):
        shell = client.get("/js/app.js").data.decode()
        assert "InterfaceExplorer" in shell
        assert "'interfaces'" in shell
        # The existing Interaction Lab must survive alongside it.
        assert "'interaction'" in shell
        assert "Interaction Lab" in shell

    def test_the_explorer_loads_no_external_script(self, client):
        """ATLAS serves its whole frontend from this process. A viewer pulled
        from a CDN would make an offline instrument need the network to draw
        an experimental structure."""
        for name in ("index.html", "js/interfaces.js", "js/structure3d.js"):
            body = client.get(f"/{name}").data.decode()
            # The SVG namespace is a URI, not a fetch; everything else that
            # looks like a URL would be one.
            body = body.replace("http://www.w3.org/2000/svg", "")
            assert "http://" not in body, name
            assert "https://" not in body, name

    def test_the_new_stylesheet_rules_are_namespaced(self, client):
        """Every selector added for this workspace is prefixed, so nothing
        here can restyle an existing one."""
        css = client.get("/components.css").data.decode()
        marker = "Interface Explorer"
        assert marker in css
        section = css[css.index(marker):]
        for line in section.splitlines():
            stripped = line.strip()
            if not stripped.startswith("."):
                continue
            selector = stripped.split("{")[0].split(",")[0].strip()
            assert selector.startswith((".iface", ".struct", ".lg-")), selector
