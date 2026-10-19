"""Tests for PK parameter extraction and mechanistic validation.

The extraction tests use verbatim label prose, because that is what the
patterns must survive. The validation tests pin the result that turns the
engine from verified into validated, and the blocking behaviour that keeps it
from inventing a parameter a label does not state.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from atlas.mechanistic.pk_extraction import (
    LabelPKRecord,
    PKParameter,
    Qualifier,
    extract_parameters,
)
from atlas.mechanistic.pk_validation import (
    AGREEMENT_TOLERANCE,
    build_parameter_set,
    solve_ka_from_tmax,
    validate_against_label,
)
from atlas.core.evidence import SourceRecord

STORE = Path(__file__).resolve().parents[1] / "data" / "pk_parameters.json"

# Verbatim from the VYLEESI (bremelanotide) label.
BREMELANOTIDE = (
    "12.3 Pharmacokinetics Following subcutaneous administration of VYLEESI, "
    "the mean plasma C max and AUC of bremelanotide are 72.8 ng/mL and 276 "
    "hr*ng/mL, respectively. Absorption Bremelanotide median T max is "
    "approximately 1.0 hour (range: 0.5 - 1.0 hours) in plasma. The absolute "
    "bioavailability of bremelanotide following subcutaneous administration of "
    "VYLEESI was about 100%. Distribution Twenty-one percent of bremelanotide "
    "binds to human serum protein. The mean (SD) volume of distribution after a "
    "single subcutaneous administration of VYLEESI is 25.0 ± 5.8 L. Elimination "
    "Following a single subcutaneous administration of VYLEESI, mean terminal "
    "half-life of bremelanotide is approximately 2.7 hours (range: 1.9– 4.0 "
    "hours) and the mean (± SD) clearance (CL/F) is 6.5 ±1.0 L/hr."
)

# Verbatim from the EGRIFTA SV (tesamorelin) label.
TESAMORELIN = (
    "12.3 Pharmacokinetics Absorption The absolute bioavailability of "
    "tesamorelin after subcutaneous administration of a 2 mg dose of EGRIFTA "
    "(1 mg/vial formulation) was determined to be less than 4% in healthy adult "
    "subjects. The median peak plasma tesamorelin concentration (T max ) was "
    "0.15 h in both populations. Distribution The mean volume of distribution "
    "(±SD) of tesamorelin following a single subcutaneous administration of the "
    "1.4 mg dose of EGRIFTA SV (2 mg/vial formulation) was 4.8 ± 1.9 L/kg in "
    "healthy subjects. Elimination Mean elimination half-life (t 1/2 ) of "
    "tesamorelin was 8 minutes in healthy subjects."
)


def by_name(params, name):
    return next((p for p in params if p.name == name), None)


class TestExtractionFromLabelProse:
    @pytest.fixture(scope="class")
    @classmethod
    def brem(cls):
        return extract_parameters(BREMELANOTIDE)

    def test_clearance_value_and_unit(self, brem):
        p = by_name(brem, "clearance")
        assert p.value == 6.5 and p.unit == "L/h" and p.sd == 1.0

    def test_volume_value_and_spread(self, brem):
        p = by_name(brem, "volume_of_distribution")
        assert p.value == 25.0 and p.unit == "L" and p.sd == 5.8

    def test_half_life_with_its_own_range(self, brem):
        p = by_name(brem, "half_life")
        assert p.value == 2.7 and p.unit == "h"
        assert (p.range_low, p.range_high) == (1.9, 4.0)

    def test_range_is_not_copied_between_parameters(self, brem):
        """Half-life and clearance share a sentence; the range belongs to one."""
        assert by_name(brem, "clearance").range_low is None

    def test_apparent_parameter_flag_is_local(self, brem):
        """CL/F appears in the same sentence as the half-life."""
        assert by_name(brem, "clearance").is_apparent
        assert not by_name(brem, "half_life").is_apparent

    def test_bioavailability_is_a_fraction(self, brem):
        p = by_name(brem, "bioavailability")
        assert p.value == pytest.approx(1.0) and p.unit == "1"

    def test_tmax_is_recorded_as_a_median(self, brem):
        p = by_name(brem, "tmax")
        assert p.value == 1.0 and p.qualifier is Qualifier.MEDIAN

    def test_paired_cmax_and_auc_both_extracted(self, brem):
        assert by_name(brem, "cmax").value == 72.8
        assert by_name(brem, "auc").value == 276.0

    def test_cmax_is_not_duplicated(self, brem):
        assert sum(1 for p in brem if p.name == "cmax") == 1

    def test_verbatim_is_a_whole_sentence_not_cut_at_a_decimal(self, brem):
        """Splitting on bare periods truncated quotes mid-number."""
        p = by_name(brem, "tmax")
        assert "0.5 - 1.0 hours" in p.verbatim
        assert not p.verbatim.endswith("0.")

    def test_every_parameter_carries_its_source_sentence(self, brem):
        assert all(len(p.verbatim) > 20 for p in brem)

    def test_units_all_parse(self, brem):
        assert all(p.dimension != "unparseable" for p in brem)


class TestBoundsAndNormalisation:
    @pytest.fixture(scope="class")
    @classmethod
    def tesa(cls):
        return extract_parameters(TESAMORELIN)

    def test_less_than_is_an_upper_bound_not_a_value(self, tesa):
        p = by_name(tesa, "bioavailability")
        assert p.qualifier is Qualifier.UPPER_BOUND
        assert not p.is_usable_as_point

    def test_weight_normalised_volume_is_flagged(self, tesa):
        p = by_name(tesa, "volume_of_distribution")
        assert p.unit == "L/kg" and p.normalised_by_weight

    def test_minutes_are_preserved_not_silently_converted(self, tesa):
        p = by_name(tesa, "half_life")
        assert p.value == 8 and p.unit == "min"

    def test_no_clearance_is_reported(self, tesa):
        assert by_name(tesa, "clearance") is None


class TestKaSolver:
    def test_round_trips_through_tmax(self):
        ke = 0.26
        for tmax in (0.25, 0.5, 1.0, 2.0):
            ka = solve_ka_from_tmax(tmax, ke)
            recovered = math.log(ka / ke) / (ka - ke)
            assert recovered == pytest.approx(tmax, rel=1e-6)

    def test_faster_absorption_gives_earlier_peak(self):
        ke = 0.26
        assert solve_ka_from_tmax(0.5, ke) > solve_ka_from_tmax(1.5, ke)

    def test_ka_always_exceeds_ke(self):
        assert solve_ka_from_tmax(1.0, 0.26) > 0.26

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            solve_ka_from_tmax(0.0, 0.26)
        with pytest.raises(ValueError):
            solve_ka_from_tmax(1.0, 0.0)


def _record(text: str, *, name="x", dose=1.75, route="subcutaneous") -> LabelPKRecord:
    return LabelPKRecord(
        generic_name=name, brand_name="B", parameters=extract_parameters(text),
        route=route, dose_mg=dose, dose_verbatim="Inject 1.75 mg",
        source=SourceRecord(source_db="openfda", source_id="test"),
    )


class TestParameterSetAssembly:
    def test_complete_label_gives_a_usable_set(self):
        ps = build_parameter_set(_record(BREMELANOTIDE))
        assert ps.is_usable
        assert set(ps.parameters) == {"CL", "V", "ka", "F"}

    def test_ke_matches_reported_half_life(self):
        """CL/V must imply the half-life the label states independently."""
        ps = build_parameter_set(_record(BREMELANOTIDE))
        ke = ps.parameters["CL"] / ps.parameters["V"]
        assert math.log(2) / ke == pytest.approx(2.7, rel=0.02)

    def test_incomplete_label_is_blocked_not_completed(self):
        ps = build_parameter_set(_record(TESAMORELIN, name="tesamorelin"))
        assert not ps.is_usable
        assert "does not report" in ps.blocked_reason

    def test_bound_bioavailability_blocks_rather_than_being_used(self):
        text = TESAMORELIN + (" The clearance was 10.0 L/hr. The volume of "
                              "distribution was 50.0 L.")
        ps = build_parameter_set(_record(text))
        assert not ps.is_usable
        assert "bound" in ps.blocked_reason

    def test_missing_dose_is_blocked(self):
        rec = _record(BREMELANOTIDE)
        rec.dose_mg = None
        assert not build_parameter_set(rec).is_usable

    def test_derived_parameters_are_declared(self):
        ps = build_parameter_set(_record(BREMELANOTIDE))
        assert "ka" in ps.derived
        assert "does not report ka directly" in ps.derived["ka"]

    def test_provenance_carries_the_source_sentences(self):
        ps = build_parameter_set(_record(BREMELANOTIDE))
        assert all(len(v) > 20 for v in ps.provenance.values())


class TestValidationAgainstObservedData:
    @pytest.fixture(scope="class")
    @classmethod
    def report(cls):
        return validate_against_label(_record(BREMELANOTIDE), n_monte_carlo=400)

    def test_three_independent_quantities_are_tested(self, report):
        assert {t.name for t in report.targets} == {"cmax", "auc", "half_life"}

    def test_auc_is_reproduced(self, report):
        t = next(t for t in report.targets if t.name == "auc")
        assert t.agrees and t.relative_error < 0.05

    def test_half_life_is_reproduced(self, report):
        t = next(t for t in report.targets if t.name == "half_life")
        assert t.agrees and t.relative_error < 0.05

    def test_cmax_is_underpredicted_and_said_so(self, report):
        """The honest finding: the point prediction misses Cmax by ~26%."""
        t = next(t for t in report.targets if t.name == "cmax")
        assert not t.agrees
        assert t.predicted < t.observed
        assert 0.15 < t.relative_error < 0.40

    def test_cmax_observation_lies_inside_the_uncertainty_interval(self, report):
        t = next(t for t in report.targets if t.name == "cmax")
        assert t.observed_within_interval is True

    def test_verdict_names_what_agreed_and_what_did_not(self, report):
        assert "auc" in report.verdict and "cmax" in report.verdict
        assert not report.all_agree

    def test_limitations_disclaim_external_validation(self, report):
        joined = " ".join(report.limitations)
        assert "not independent external validation" in joined
        assert "one-compartment" in joined

    def test_blocked_record_validates_to_nothing(self):
        r = validate_against_label(_record(TESAMORELIN), n_monte_carlo=50)
        assert r.targets == []
        assert "does not report" in " ".join(r.limitations)

    def test_tolerance_is_documented(self):
        assert 0.05 <= AGREEMENT_TOLERANCE <= 0.5


@pytest.mark.skipif(not STORE.exists(), reason="PK store not built")
class TestParameterStore:
    @pytest.fixture(scope="class")
    @classmethod
    def store(cls):
        return json.loads(STORE.read_text())

    def test_store_is_strict_json(self):
        def reject(c):
            raise ValueError(f"bare {c}")
        json.loads(STORE.read_text(), parse_constant=reject)

    def test_only_fully_parameterised_compounds_are_usable(self, store):
        for key in store["usable_keys"]:
            params = store["entries"][key]["parameters"]
            assert set(params) >= {"CL", "V", "ka", "F"}

    def test_pt141_is_usable_and_validated(self, store):
        assert "pt-141" in store["usable_keys"]
        v = store["entries"]["pt-141"]["validation"]
        assert v["n_agreeing"] >= 2

    def test_insufficient_entries_state_what_is_missing(self, store):
        blocked = [e for e in store["entries"].values()
                   if e["status"] == "insufficient_parameters"]
        assert blocked
        assert all("does not report" in e["blocked_reason"] for e in blocked)

    def test_every_usable_entry_carries_provenance_sentences(self, store):
        for key in store["usable_keys"]:
            prov = store["entries"][key]["provenance"]
            assert set(prov) >= {"clearance", "volume_of_distribution",
                                 "bioavailability", "tmax"}
            assert all(len(v) > 20 for v in prov.values())

    def test_store_does_not_overclaim_coverage(self, store):
        assert store["n_usable"] <= store["n_with_label"] <= store["n_attempted"]


@pytest.mark.skipif(not STORE.exists(), reason="PK store not built")
class TestGateIntegration:
    @pytest.fixture(scope="class")
    @classmethod
    def client(cls):
        from atlas.api.app import create_app
        app = create_app()
        app.config.update(TESTING=True)
        return app.test_client()

    def test_human_pk_simulation_is_now_permitted_for_pt141(self, client):
        rows = client.get("/api/registry").get_json()["rows"]
        row = next(r for r in rows if r["key"] == "pt-141")
        assert "human_pk_simulation" in row["permitted"]

    def test_it_remains_refused_where_parameters_are_absent(self, client):
        rows = client.get("/api/registry").get_json()["rows"]
        for key in ("bpc-157", "semaglutide", "tesamorelin"):
            row = next(r for r in rows if r["key"] == key)
            assert "human_pk_simulation" not in row["permitted"]

    def test_simulation_for_pt141_is_class_b_without_abstention(self, client):
        store = json.loads(STORE.read_text())
        e = store["entries"]["pt-141"]
        r = client.post("/api/simulate", json={
            "model": "one_compartment_oral", "parameters": e["parameters"],
            "dose_mg": e["dose_mg"], "compound_key": "pt-141",
        }).get_json()
        assert r["evidence_class"].startswith("B_")
        assert r["abstention"] is None
        assert "FDA label" in r["interpretation"]

    def test_simulation_elsewhere_still_abstains(self, client):
        r = client.post("/api/simulate", json={
            "model": "one_compartment_oral",
            "parameters": {"CL": 1.0, "V": 10.0, "ka": 1.5, "F": 0.8},
            "dose_mg": 10.0, "compound_key": "bpc-157",
        }).get_json()
        assert r["evidence_class"].startswith("D_")
        assert r["abstention"] is not None

    def test_endpoint_reports_store_coverage(self, client):
        pk = client.get("/api/pk-parameters").get_json()
        assert pk["n_usable"] >= 1
        assert pk["n_usable"] < pk["n_attempted"]
