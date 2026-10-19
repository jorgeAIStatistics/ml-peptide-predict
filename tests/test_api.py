"""Tests for the JSON API.

The gating tests matter most: the API is the boundary where a solver run could
be mislabelled as a statement about a compound, so the labelling is pinned here
as well as in the engine.
"""

from __future__ import annotations

import pytest

from atlas.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


BASE_ORAL = {
    "model": "one_compartment_oral",
    "parameters": {"CL": 1.0, "V": 10.0, "ka": 1.5, "F": 0.8},
    "dose_mg": 10.0,
}


class TestMeta:
    def test_meta_reports_a_snapshot(self, client):
        m = client.get("/api/meta").get_json()
        assert m["snapshot_id"]
        assert m["n_entries"] > 0

    def test_scope_note_disclaims_clinical_use(self, client):
        note = client.get("/api/meta").get_json()["scope_note"].lower()
        assert "does not provide dosing" in note

    def test_human_pk_is_not_permitted_for_any_entry(self, client):
        """No PK parameters have been extracted, so the honest count is zero."""
        counts = client.get("/api/meta").get_json()["permitted_counts"]
        assert counts["human_pk_simulation"] == 0
        assert counts["quantitative_safety_prediction"] == 0


class TestRegistry:
    def test_registry_returns_rows(self, client):
        data = client.get("/api/registry").get_json()
        assert data["n"] == len(data["rows"]) > 0

    def test_search_matches_an_alias_not_only_the_name(self, client):
        data = client.get("/api/registry?q=thymalfasin").get_json()
        assert any(r["key"] == "thymosin-alpha-1" for r in data["rows"])

    def test_filter_by_identity_status(self, client):
        data = client.get("/api/registry?identity_status=mixture_not_a_single_molecular_entity").get_json()
        assert data["n"] > 0
        assert all(r["identity_status"].startswith("mixture") for r in data["rows"])

    def test_filter_by_area(self, client):
        data = client.get("/api/registry?area=dermal_and_copper_associated").get_json()
        assert all(r["research_area"] == "dermal_and_copper_associated" for r in data["rows"])


class TestCompoundDetail:
    def test_unknown_key_is_404(self, client):
        assert client.get("/api/compound/not-a-compound").status_code == 404

    def test_fields_carry_their_own_source(self, client):
        """Identity is assembled from several databases; each field says which."""
        d = client.get("/api/compound/semaglutide").get_json()
        mw = d["identity_fields"]["molecular_weight_g_per_mol"]
        assert mw["source_db"] and mw["source_id"]
        assert d["identity_fields"]["chembl_id"]["source_db"] == "chembl"

    def test_kisspeptin_reports_its_ambiguity(self, client):
        d = client.get("/api/compound/kisspeptin").get_json()
        assert d["identity_status"].startswith("ambiguous")
        assert any("disagrees" in c for c in d["caveats"])

    def test_mixture_has_no_structure_and_is_refused(self, client):
        d = client.get("/api/compound/glow").get_json()
        assert d["smiles"] is None
        assert "structure_based_prediction" in d["refused"]

    def test_sources_carry_retrieval_provenance(self, client):
        d = client.get("/api/compound/semaglutide").get_json()
        assert d["sources"]
        assert all(s.get("retrieved_at") for s in d["sources"])


class TestModels:
    def test_every_model_exposes_equations_and_units(self, client):
        models = client.get("/api/models").get_json()["models"]
        assert len(models) >= 3
        for m in models:
            assert m["equations"], m["name"]
            assert all(p["unit"] for p in m["parameters"])


class TestSimulate:
    def test_runs_and_returns_a_trajectory(self, client):
        r = client.post("/api/simulate", json=BASE_ORAL)
        assert r.status_code == 200
        d = r.get_json()
        assert len(d["times_h"]) == len(d["concentration_mg_per_l"]) > 10
        assert d["run_id"]

    def test_nca_recovers_the_analytic_auc(self, client):
        """Independent check: NCA on the trajectory must match F*Dose/CL."""
        d = client.post("/api/simulate", json=BASE_ORAL).get_json()
        exact = d["nca"]["analytic_auc_inf_mg_h_per_l"]
        got = d["nca"]["auc_inf_mg_h_per_l"]
        assert exact == pytest.approx(8.0)
        assert got == pytest.approx(exact, rel=1e-3)

    def test_mass_balance_is_reported_and_tight(self, client):
        d = client.post("/api/simulate", json=BASE_ORAL).get_json()
        assert d["diagnostics"]["mass_balance_relative_error"] < 1e-9

    def test_repeated_dosing_scales_exposure(self, client):
        d = client.post("/api/simulate", json={**BASE_ORAL, "n_doses": 4}).get_json()
        assert d["n_doses"] == 4
        assert d["nca"]["analytic_auc_inf_mg_h_per_l"] == pytest.approx(32.0)

    def test_flip_flop_is_flagged(self, client):
        d = client.post("/api/simulate", json={
            "model": "one_compartment_oral",
            "parameters": {"CL": 2.0, "V": 10.0, "ka": 0.05, "F": 1.0},
            "dose_mg": 10.0,
        }).get_json()
        assert d["diagnostics"]["is_flip_flop"] is True

    def test_states_sum_to_the_dose(self, client):
        d = client.post("/api/simulate", json=BASE_ORAL).get_json()
        totals = [sum(vals[i] for vals in d["states"].values())
                  for i in range(len(d["times_h"]))]
        assert all(t == pytest.approx(10.0, rel=1e-6) for t in totals)


class TestSimulateValidation:
    def test_missing_parameter_is_400_with_the_name(self, client):
        r = client.post("/api/simulate", json={
            "model": "one_compartment_iv", "parameters": {"CL": 1.0}, "dose_mg": 10.0,
        })
        assert r.status_code == 400
        assert "V" in r.get_json()["error"]

    def test_out_of_range_bioavailability_is_rejected(self, client):
        r = client.post("/api/simulate", json={
            **BASE_ORAL, "parameters": {**BASE_ORAL["parameters"], "F": 2.0},
        })
        assert r.status_code == 400
        assert r.get_json()["error_kind"] == "parameter"

    def test_unknown_model_is_rejected(self, client):
        r = client.post("/api/simulate", json={**BASE_ORAL, "model": "telepathy"})
        assert r.status_code == 400

    def test_negative_dose_is_rejected(self, client):
        r = client.post("/api/simulate", json={**BASE_ORAL, "dose_mg": -1})
        assert r.status_code == 400

    def test_excessive_output_points_rejected(self, client):
        r = client.post("/api/simulate", json={**BASE_ORAL, "n_points": 10_000_000})
        assert r.status_code == 400

    def test_excessive_dose_count_rejected(self, client):
        r = client.post("/api/simulate", json={**BASE_ORAL, "n_doses": 100_000})
        assert r.status_code == 400


class TestSimulationLabelling:
    """A solver will integrate anything; what it may be *called* is gated."""

    def test_unattributed_run_is_exploratory(self, client):
        d = client.post("/api/simulate", json=BASE_ORAL).get_json()
        assert d["evidence_class"].startswith("D_")
        assert d["abstention"] is None

    def test_attributed_run_is_still_exploratory_and_carries_an_abstention(self, client):
        d = client.post("/api/simulate", json={**BASE_ORAL, "compound_key": "bpc-157"}).get_json()
        assert d["evidence_class"].startswith("D_")
        assert d["abstention"] is not None
        assert "not a statement about BPC-157 in humans" in d["abstention"]["explanation"]

    def test_abstention_names_the_measured_evidence(self, client):
        d = client.post("/api/simulate", json={**BASE_ORAL, "compound_key": "bpc-157"}).get_json()
        assert "trials with posted results" in d["abstention"]["explanation"]
        assert d["abstention"]["what_would_resolve_it"]

    def test_mixture_attribution_also_abstains(self, client):
        d = client.post("/api/simulate", json={**BASE_ORAL, "compound_key": "glow"}).get_json()
        assert d["abstention"] is not None

    def test_unknown_compound_key_does_not_crash(self, client):
        r = client.post("/api/simulate", json={**BASE_ORAL, "compound_key": "nope"})
        assert r.status_code == 200
        assert r.get_json()["evidence_class"].startswith("D_")


class TestVerificationEndpoint:
    def test_all_solver_checks_pass(self, client):
        v = client.get("/api/verification").get_json()
        assert v["all_passed"] is True
        assert v["n_passed"] == v["n_checks"] > 50

    def test_note_separates_verification_from_validation(self, client):
        note = client.get("/api/verification").get_json()["note"]
        assert "numerical correctness only" in note
        assert "not evidence" in note


class TestStaticFrontend:
    @pytest.mark.parametrize("path", ["/", "/app.css", "/app.js"])
    def test_assets_are_served(self, client, path):
        assert client.get(path).status_code == 200

    def test_page_has_no_external_script_dependency(self, client):
        """The interface must work offline: no CDN scripts or stylesheets."""
        html = client.get("/").get_data(as_text=True)
        assert "http://" not in html.replace("http://www.w3.org", "")
        assert "cdn" not in html.lower()
