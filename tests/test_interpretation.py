"""Tests for the scientific interpretation engine.

The property that matters is that interpretations are *computed* from the value
and the producing model's metadata, not attached as prewritten narrative. These
tests check the arithmetic, and check that the "does not establish" statements
are present for the quantities most often over-read.
"""

from __future__ import annotations

import math

import pytest

from atlas.interpretation.definitions import DEFINITIONS, get_definition
from atlas.interpretation.engine import (
    concentration_vs_affinity,
    interpret,
)


def derived(interp, label_fragment):
    return next((d for d in interp.derived if label_fragment.lower() in d["label"].lower()),
                None)


class TestDefinitions:
    def test_every_definition_states_what_it_cannot_establish(self):
        for key, d in DEFINITIONS.items():
            assert d.does_not_establish, f"{key} has no negative statements"

    def test_every_definition_offers_a_validating_measurement(self):
        for key, d in DEFINITIONS.items():
            assert d.validating_measurements, key

    def test_affinity_definition_refuses_the_efficacy_reading(self):
        joined = " ".join(get_definition("pki").does_not_establish).lower()
        assert "functional response" in joined
        assert "clinical" in joined

    def test_exposure_definition_refuses_the_benefit_reading(self):
        joined = " ".join(get_definition("auc").does_not_establish).lower()
        assert "benefit" in joined or "outcome" in joined

    def test_half_life_separates_drug_from_effect(self):
        joined = " ".join(get_definition("half_life").does_not_establish).lower()
        assert "duration of biological effect" in joined

    def test_occupancy_definition_mentions_receptor_reserve(self):
        joined = " ".join(get_definition("receptor_occupancy").does_not_establish).lower()
        assert "reserve" in joined

    def test_unknown_endpoint_returns_none(self):
        assert get_definition("not_a_real_endpoint") is None


class TestPkiArithmetic:
    def test_directive_example_reproduced_exactly(self):
        """pKi 5.45 must give Ki ~3.55 uM."""
        i = interpret("pki", 5.45)
        d = derived(i, "Equivalent Ki")
        assert "3.55 uM" in d["value"]

    def test_ki_conversion_is_exact_across_magnitudes(self):
        for pki, expected in ((9.0, "1 nM"), (6.0, "1 uM"), (7.68, "20.9 nM")):
            d = derived(interpret("pki", pki), "Equivalent Ki")
            assert expected in d["value"], (pki, d["value"])

    def test_one_unit_is_described_as_tenfold(self):
        d = derived(interpret("pki", 7.0), "one unit")
        assert "tenfold" in d["value"]

    def test_interval_becomes_a_fold_range(self):
        i = interpret("pki", 7.0, {"interval_low": 6.0, "interval_high": 8.0})
        d = derived(i, "Interval")
        assert "100-fold" in d["value"]

    def test_no_interval_means_no_interval_derivation(self):
        assert derived(interpret("pki", 7.0), "Interval") is None

    def test_round_trip_through_ki(self):
        i = interpret("ki", 20.9)
        assert float(derived(i, "Equivalent pKi")["value"]) == pytest.approx(7.68, abs=0.01)


class TestPkArithmetic:
    def test_half_life_gives_rate_constant(self):
        d = derived(interpret("half_life", 2.7), "rate constant")
        assert float(d["value"].split()[0]) == pytest.approx(math.log(2) / 2.7, rel=1e-2)

    def test_time_to_steady_state_is_about_four_half_lives(self):
        d = derived(interpret("half_life", 10.0), "steady state")
        assert 40 <= float(d["value"].split()[0]) <= 45

    def test_accumulation_ratio_is_one_when_dosing_is_sparse(self):
        """A 24 h interval on a 2.7 h half-life leaves nothing to accumulate."""
        d = derived(interpret("half_life", 2.7, {"dosing_interval_h": 24}),
                    "Accumulation")
        assert float(d["value"].rstrip("x")) == pytest.approx(1.0, abs=0.01)

    def test_accumulation_ratio_is_two_when_dosing_every_half_life(self):
        d = derived(interpret("half_life", 10.0, {"dosing_interval_h": 10}),
                    "Accumulation")
        assert float(d["value"].rstrip("x")) == pytest.approx(2.0, abs=0.01)

    def test_auc_implies_clearance(self):
        """CL = F*Dose/AUC; 1.75 mg over 276 ng*h/mL is about 6.3 L/h."""
        i = interpret("auc", 276.0, {"dose_mg": 1.75, "bioavailability": 1.0})
        d = derived(i, "clearance")
        assert float(d["value"].split()[0]) == pytest.approx(6.34, rel=0.02)

    def test_missing_bioavailability_is_declared_not_assumed_silently(self):
        i = interpret("auc", 276.0, {"dose_mg": 1.75})
        assert "assumed 1" in derived(i, "clearance")["how"]

    def test_cmax_converts_to_molar_with_a_weight(self):
        i = interpret("cmax", 72.8, {"molecular_weight": 1025.2})
        assert "71 nM" in derived(i, "Molar")["value"]

    def test_cmax_without_weight_cannot_convert(self):
        assert derived(interpret("cmax", 72.8), "Molar") is None


class TestPerformanceMetrics:
    def test_mae_becomes_a_fold_error(self):
        d = derived(interpret("mae", 1.0), "fold-change")
        assert "10-fold" in d["value"]

    def test_mae_compared_against_the_noise_floor(self):
        i = interpret("mae", 0.685, {"experimental_noise_floor": 0.055})
        d = derived(i, "noise floor")
        assert float(d["value"].split("x")[0]) == pytest.approx(12.5, rel=0.05)

    def test_coverage_gap_is_signed_and_judged(self):
        i = interpret("coverage", 0.838, {"nominal_coverage": 0.90})
        d = derived(i, "nominal")
        assert d["value"].startswith("-")
        assert "too narrow" in d["how"]

    def test_adequate_coverage_is_described_as_such(self):
        i = interpret("coverage", 0.909, {"nominal_coverage": 0.90})
        assert "within 3 points" in derived(i, "nominal")["how"]

    def test_mae_warns_about_comparing_across_splits(self):
        joined = " ".join(interpret("mae", 0.5).biological_meaning["common_misreadings"])
        assert "random split" in joined


class TestExposureAffinityBridge:
    def test_pt141_occupancy_is_computed_from_measured_values(self):
        r = concentration_vs_affinity(72.8, 7.68, 1025.2, unbound_fraction=0.79)
        assert r["ratio_to_ki"] == pytest.approx(2.69, rel=0.05)
        assert 0.70 < r["implied_occupancy_at_peak"] < 0.76

    def test_unbound_fraction_lowers_the_ratio(self):
        total = concentration_vs_affinity(72.8, 7.68, 1025.2)
        free = concentration_vs_affinity(72.8, 7.68, 1025.2, unbound_fraction=0.5)
        assert free["ratio_to_ki"] < total["ratio_to_ki"]

    def test_occupancy_follows_the_equilibrium_model(self):
        """At a concentration equal to Ki, occupancy is one half."""
        ki_molar = 10 ** (-7.0)
        mw = 1000.0
        cmax_ng_ml = ki_molar * mw * 1e9 / 1e3
        r = concentration_vs_affinity(cmax_ng_ml, 7.0, mw)
        assert r["implied_occupancy_at_peak"] == pytest.approx(0.5, abs=0.01)

    def test_bridge_always_carries_its_caveats(self):
        r = concentration_vs_affinity(72.8, 7.68, 1025.2)
        joined = " ".join(r["caveats"]).lower()
        assert "not target-tissue" in joined
        assert "not a functional response" in joined


class TestInterpretationStructure:
    def test_all_six_sections_are_populated(self):
        i = interpret("pki", 7.68, {
            "target": "MC4R", "species": "human",
            "contributions": [{"kind": "measurement", "detail": "x"}],
            "assumptions": ["y"],
        })
        assert i.what_is_measured["definition"]
        assert i.mathematics["transformation"]
        assert i.biological_meaning["does_not_establish"]
        assert i.uncertainty["drivers"]
        assert i.why_this_result
        assert i.validation["measurements_that_would_test_it"]

    def test_dominant_driver_is_listed_first(self):
        i = interpret("pki", 7.0, {"dominant_uncertainty": "THE REASON"})
        assert i.uncertainty["drivers"][0] == "THE REASON"
        assert i.uncertainty["dominant_driver"] == "THE REASON"

    def test_context_metadata_is_carried_through(self):
        i = interpret("pki", 7.0, {"target": "MC4R", "species": "human",
                                   "source_kind": "measurement"})
        assert i.what_is_measured["target"] == "MC4R"
        assert i.what_is_measured["source_kind"] == "measurement"

    def test_unknown_endpoint_returns_none(self):
        assert interpret("nonsense", 1.0) is None

    def test_none_value_produces_no_derived_quantities(self):
        i = interpret("pki", None)
        assert i.derived == []
        assert i.mathematics["stated_value"] is None

    def test_serialises_to_plain_data(self):
        d = interpret("pki", 7.0).as_dict()
        assert set(d) >= {"endpoint", "symbol", "value", "derived",
                          "biological_meaning", "uncertainty", "validation"}


class TestAttachedToRealResults:
    @pytest.fixture(scope="class")
    @classmethod
    def client(cls):
        from atlas.api.app import create_app
        app = create_app()
        app.config.update(TESTING=True)
        return app.test_client()

    def test_measurement_carries_interpretations(self, client):
        r = client.post("/api/predict", json={"compound_key": "pt-141"}).get_json()
        symbols = {i["symbol"] for i in r["interpretations"]}
        assert "pKi" in symbols and "Ki" in symbols

    def test_measured_interpretation_cites_the_real_study_count(self, client):
        r = client.post("/api/predict", json={"compound_key": "pt-141"}).get_json()
        pki = next(i for i in r["interpretations"] if i["symbol"] == "pKi")
        detail = " ".join(c["detail"] for c in pki["why_this_result"])
        assert "independent studies" in detail
        assert "CHEMBL" in detail

    def test_discordant_measurement_names_that_as_the_dominant_uncertainty(self, client):
        r = client.post("/api/predict", json={"compound_key": "pt-141"}).get_json()
        pki = next(i for i in r["interpretations"] if i["symbol"] == "pKi")
        assert "disagree" in (pki["uncertainty"]["dominant_driver"] or "")

    def test_abstention_carries_no_fabricated_interpretation(self, client):
        r = client.post("/api/predict", json={"compound_key": "pinealon"}).get_json()
        assert r["answer_kind"] == "abstention"
        assert r["interpretations"] == []

    def test_simulation_interprets_its_derived_quantities(self, client):
        import json
        from pathlib import Path
        store = json.loads(
            (Path(__file__).resolve().parents[1] / "data" / "pk_parameters.json").read_text())
        e = store["entries"]["pt-141"]
        r = client.post("/api/simulate", json={
            "model": "one_compartment_oral", "parameters": e["parameters"],
            "dose_mg": e["dose_mg"], "compound_key": "pt-141",
        }).get_json()
        symbols = {i["symbol"] for i in r["interpretations"]}
        assert {"Cmax", "AUC", "t½"} <= symbols

    def test_exploratory_simulation_says_the_parameters_are_assumed(self, client):
        r = client.post("/api/simulate", json={
            "model": "one_compartment_oral",
            "parameters": {"CL": 1.0, "V": 10.0, "ka": 1.5, "F": 0.8},
            "dose_mg": 10.0,
        }).get_json()
        auc = next(i for i in r["interpretations"] if i["symbol"] == "AUC")
        joined = " ".join(auc["validation"]["open_assumptions"])
        assert "supplied by the user" in joined

    def test_model_card_interprets_its_own_metrics(self, client):
        r = client.get("/api/model-card").get_json()
        symbols = {i["symbol"] for i in r["interpretations"]}
        assert {"MAE", "coverage"} <= symbols

    def test_mae_interpretation_contrasts_the_random_split(self, client):
        r = client.get("/api/model-card").get_json()
        mae = next(i for i in r["interpretations"] if i["symbol"] == "MAE")
        assert "random split" in (mae["uncertainty"]["dominant_driver"] or "")
