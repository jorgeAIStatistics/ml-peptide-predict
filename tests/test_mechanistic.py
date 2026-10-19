"""Tests for the mechanistic PK engine.

Numerical results are checked against closed-form mathematics rather than
against stored output, so these tests fail if the solver drifts rather than
merely if its output changes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atlas.mechanistic.metrics import (
    auc_linear,
    auc_linlog,
    cmax_tmax,
    nca,
    terminal_slope,
)
from atlas.mechanistic.models import (
    MODEL_REGISTRY,
    OneCompartmentIV,
    OneCompartmentOral,
    PKModel,
    ParameterError,
    ParameterSpec,
    TwoCompartmentIV,
    get_model,
)
from atlas.mechanistic.solver import (
    DoseEvent,
    SolverError,
    SolverSettings,
    simulate,
)
from atlas.mechanistic.verify import REFERENCE_CASES, verify_model

IV = {"CL": 1.0, "V": 10.0}
ORAL = {"CL": 1.0, "V": 10.0, "ka": 1.5, "F": 0.8}
TWO = {"CL": 1.0, "V1": 5.0, "Q": 2.0, "V2": 20.0}


class TestParameterSpecs:
    def test_units_carry_correct_dimensions(self):
        m = OneCompartmentIV()
        assert m.spec("CL").dimension == __import__(
            "atlas.core.units", fromlist=["CLEARANCE"]
        ).CLEARANCE
        assert m.spec("V").dimension.length == 3

    def test_missing_parameter_rejected(self):
        with pytest.raises(ParameterError, match="missing parameters"):
            OneCompartmentIV().validate({"CL": 1.0})

    def test_unexpected_parameter_rejected(self):
        """A stray ka on an IV model probably means the wrong model was chosen."""
        with pytest.raises(ParameterError, match="unexpected parameters"):
            OneCompartmentIV().validate({"CL": 1.0, "V": 10.0, "ka": 1.0})

    def test_nonpositive_clearance_rejected(self):
        with pytest.raises(ParameterError, match="must be > 0"):
            OneCompartmentIV().validate({"CL": 0.0, "V": 10.0})

    def test_bioavailability_above_one_rejected(self):
        with pytest.raises(ParameterError, match="must be <= 1"):
            OneCompartmentOral().validate({**ORAL, "F": 1.5})

    def test_nonfinite_rejected(self):
        with pytest.raises(ParameterError, match="finite"):
            OneCompartmentIV().validate({"CL": float("nan"), "V": 10.0})

    def test_zero_bioavailability_allowed_when_inclusive(self):
        spec = ParameterSpec("F", "1", "frac", lower=0.0, upper=1.0, lower_inclusive=True)
        spec.check(0.0)

    def test_registry_lookup(self):
        assert isinstance(get_model("one_compartment_iv"), OneCompartmentIV)
        with pytest.raises(KeyError, match="unknown model"):
            get_model("three_compartment_telepathy")

    def test_every_registered_model_declares_units_for_all_parameters(self):
        for cls in MODEL_REGISTRY.values():
            model = cls()
            for spec in model.parameters:
                assert spec.unit, f"{model.name}.{spec.name} has no unit"
                assert spec.description.endswith(".")


class TestAnalyticalSolutions:
    def test_iv_initial_concentration_is_dose_over_volume(self):
        m = OneCompartmentIV()
        c = m.analytical_concentration(np.array([0.0]), 10.0, IV)
        assert c[0] == pytest.approx(1.0)

    def test_iv_half_life(self):
        m = OneCompartmentIV()
        th = m.terminal_half_life_h(IV)
        c = m.analytical_concentration(np.array([0.0, th]), 10.0, IV)
        assert c[1] / c[0] == pytest.approx(0.5)

    def test_oral_starts_at_zero(self):
        m = OneCompartmentOral()
        assert m.analytical_concentration(np.array([0.0]), 10.0, ORAL)[0] == 0.0

    def test_oral_tmax_matches_numerical_maximum(self):
        m = OneCompartmentOral()
        tmax = m.analytical_tmax_h(ORAL)
        grid = np.linspace(0, 10 * tmax, 200001)
        c = m.analytical_concentration(grid, 10.0, ORAL)
        assert grid[int(np.argmax(c))] == pytest.approx(tmax, rel=1e-3)

    def test_oral_cmax_is_the_peak(self):
        m = OneCompartmentOral()
        cmax = m.analytical_cmax(10.0, ORAL)
        grid = np.linspace(0, 100, 100001)
        assert float(np.max(m.analytical_concentration(grid, 10.0, ORAL))) == pytest.approx(
            cmax, rel=1e-6
        )

    def test_ka_equals_ke_singularity_is_finite(self):
        """ka == ke makes the Bateman denominator vanish; the limit is finite."""
        m = OneCompartmentOral()
        p = {"CL": 1.0, "V": 10.0, "ka": 0.1, "F": 1.0}  # ke = 0.1 exactly
        c = m.analytical_concentration(np.linspace(0, 100, 501), 10.0, p)
        assert np.all(np.isfinite(c))
        assert np.max(c) > 0

    def test_ka_equals_ke_limit_matches_nearby_value(self):
        """The limit must agree with the general form evaluated just off it."""
        m = OneCompartmentOral()
        t = np.linspace(0.1, 60, 200)
        at_limit = m.analytical_concentration(t, 10.0, {"CL": 1.0, "V": 10.0, "ka": 0.1, "F": 1.0})
        nearby = m.analytical_concentration(
            t, 10.0, {"CL": 1.0, "V": 10.0, "ka": 0.1 * (1 + 1e-6), "F": 1.0}
        )
        assert np.allclose(at_limit, nearby, rtol=1e-4)

    def test_flip_flop_detected_when_absorption_is_slower(self):
        m = OneCompartmentOral()
        assert m.is_flip_flop({"CL": 2.0, "V": 10.0, "ka": 0.05, "F": 1.0})
        assert not m.is_flip_flop(ORAL)

    def test_flip_flop_terminal_slope_reflects_absorption(self):
        """In flip-flop kinetics the terminal half-life is the absorption one."""
        m = OneCompartmentOral()
        p = {"CL": 2.0, "V": 10.0, "ka": 0.05, "F": 1.0}
        assert m.terminal_half_life_h(p) == pytest.approx(math.log(2) / 0.05)

    def test_two_compartment_hybrid_constants_are_real_and_ordered(self):
        m = TwoCompartmentIV()
        alpha, beta = m._hybrid(TWO)
        assert alpha > beta > 0

    def test_two_compartment_is_biexponential_not_monoexponential(self):
        """A second phase must be visible: log-concentration cannot be linear."""
        m = TwoCompartmentIV()
        t = np.linspace(0.1, 60, 300)
        logc = np.log(m.analytical_concentration(t, 10.0, TWO))
        slope, intercept = np.polyfit(t, logc, 1)
        resid = logc - (slope * t + intercept)
        assert float(np.max(np.abs(resid))) > 0.05

    def test_steady_state_volume(self):
        assert TwoCompartmentIV().steady_state_volume_l(TWO) == pytest.approx(25.0)


class TestSolver:
    def test_numerical_matches_analytical_for_every_reference_case(self):
        for name, cases in REFERENCE_CASES.items():
            model = MODEL_REGISTRY[name]()
            for p in cases:
                th = model.terminal_half_life_h(p)
                t = np.linspace(0, 8 * th, 300)
                exact = model.analytical_concentration(t, 10.0, p)
                comp = "depot_mg" if "depot_mg" in model.state_names else "central_mg"
                res = simulate(
                    model, p, [DoseEvent(0.0, 10.0, comp)], t,
                    settings=SolverSettings(rtol=1e-11, atol=1e-14),
                )
                scale = float(np.max(np.abs(exact)))
                err = float(np.max(np.abs(res.concentration_mg_per_l - exact))) / scale
                assert err < 1e-8, f"{name} {p}: relative error {err:.2e}"

    def test_mass_balance_is_reported(self):
        res = simulate(
            OneCompartmentIV(), IV, [DoseEvent(0.0, 10.0, "central_mg")],
            np.linspace(0, 50, 100),
        )
        assert res.mass_balance_relative_error < 1e-9

    def test_repeated_dosing_accumulates(self):
        m = OneCompartmentIV()
        tau = m.terminal_half_life_h(IV)
        doses = [DoseEvent(i * tau, 10.0, "central_mg") for i in range(5)]
        t = np.array([d.time_h for d in doses])
        res = simulate(m, IV, doses, t)
        peaks = res.concentration_mg_per_l
        assert np.all(np.diff(peaks) > 0), "trough-to-peak should accumulate"

    def test_repeated_dosing_approaches_expected_accumulation_ratio(self):
        """Dosing every half-life approaches a twofold accumulation ratio."""
        m = OneCompartmentIV()
        tau = m.terminal_half_life_h(IV)
        doses = [DoseEvent(i * tau, 10.0, "central_mg") for i in range(30)]
        t = np.array([29 * tau])
        res = simulate(m, IV, doses, t)
        assert res.concentration_mg_per_l[0] == pytest.approx(2.0, rel=1e-3)

    def test_unknown_compartment_rejected(self):
        with pytest.raises(ParameterError, match="no compartment"):
            simulate(
                OneCompartmentIV(), IV, [DoseEvent(0.0, 10.0, "spleen")],
                np.linspace(0, 10, 10),
            )

    def test_no_dose_rejected(self):
        with pytest.raises(ValueError, match="at least one dose"):
            simulate(OneCompartmentIV(), IV, [], np.linspace(0, 10, 10))

    def test_negative_dose_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            DoseEvent(0.0, -5.0, "central_mg")

    def test_decreasing_times_rejected(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            simulate(
                OneCompartmentIV(), IV, [DoseEvent(0.0, 10.0, "central_mg")],
                np.array([0.0, 5.0, 2.0]),
            )

    def test_dose_after_first_output_time_rejected(self):
        with pytest.raises(ValueError, match="first dose"):
            simulate(
                OneCompartmentIV(), IV, [DoseEvent(5.0, 10.0, "central_mg")],
                np.array([0.0, 10.0]),
            )

    def test_run_id_is_deterministic_and_parameter_sensitive(self):
        args = (OneCompartmentIV(), IV, [DoseEvent(0.0, 10.0, "central_mg")],
                np.linspace(0, 10, 50))
        a = simulate(*args).run_id
        b = simulate(*args).run_id
        c = simulate(OneCompartmentIV(), {"CL": 1.1, "V": 10.0},
                     [DoseEvent(0.0, 10.0, "central_mg")], np.linspace(0, 10, 50)).run_id
        assert a == b and a != c

    def test_concentration_unit_conversion(self):
        res = simulate(
            OneCompartmentIV(), IV, [DoseEvent(0.0, 10.0, "central_mg")],
            np.array([0.0]),
        )
        assert res.concentration_ng_per_ml()[0] == pytest.approx(1000.0)

    def test_state_lookup_by_name(self):
        dose, t_end = 10.0, 100.0
        res = simulate(
            OneCompartmentIV(), IV, [DoseEvent(0.0, dose, "central_mg")],
            np.array([0.0, t_end]),
        )
        # Exact: eliminated(t) = D * (1 - exp(-ke t)). At 100 h roughly 4.5e-5
        # of the dose is still present, so the total is not yet 10 mg.
        ke = IV["CL"] / IV["V"]
        expected = dose * (1.0 - math.exp(-ke * t_end))
        assert res.state("eliminated_mg")[-1] == pytest.approx(expected, rel=1e-9)
        with pytest.raises(KeyError):
            res.state("liver_mg")


class LeakyModel(PKModel):
    """A deliberately non-conservative model used to prove the guard fires."""

    name = "leaky_test_model"
    description = "Loses drug without accounting for it."
    state_names = ("central_mg", "eliminated_mg")
    central_index = 0
    parameters = (ParameterSpec("CL", "L/h", "Clearance."), ParameterSpec("V", "L", "Volume."))

    def rhs(self, t, y, p):
        out = p["CL"] / p["V"] * y[0]
        # Only half the eliminated drug is recorded: mass vanishes.
        return np.array([-out, 0.5 * out])

    def initial_state(self, dose_mg, p):
        return np.array([float(dose_mg), 0.0])

    def central_volume(self, p):
        return p["V"]


class TestMassBalanceGuard:
    def test_non_conservative_model_is_rejected(self):
        with pytest.raises(SolverError, match="mass balance violated"):
            simulate(
                LeakyModel(), IV, [DoseEvent(0.0, 10.0, "central_mg")],
                np.linspace(0, 50, 100),
            )

    def test_guard_can_be_relaxed_deliberately(self):
        res = simulate(
            LeakyModel(), IV, [DoseEvent(0.0, 10.0, "central_mg")],
            np.linspace(0, 50, 100), mass_balance_rtol=1.0,
        )
        assert res.mass_balance_relative_error > 0.1


class TestMetrics:
    def test_logtrapezoid_is_exact_for_monoexponential(self):
        """The log rule integrates an exponential exactly; the linear rule does not."""
        k, c0, T = 0.3, 100.0, 24.0
        t = np.linspace(0, T, 13)
        c = c0 * np.exp(-k * t)
        exact = c0 / k * (1 - math.exp(-k * T))
        assert auc_linlog(t, c) == pytest.approx(exact, rel=1e-12)
        assert auc_linear(t, c) > exact

    def test_linear_rule_overestimates_convex_decline(self):
        k, c0 = 0.5, 10.0
        t = np.linspace(0, 20, 9)
        c = c0 * np.exp(-k * t)
        assert auc_linear(t, c) / auc_linlog(t, c) > 1.01

    def test_cmax_tmax_on_a_known_peak(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        c = np.array([0.0, 5.0, 9.0, 4.0])
        assert cmax_tmax(t, c) == (9.0, 2.0)

    def test_terminal_slope_recovers_known_rate(self):
        k = 0.123
        t = np.linspace(5, 50, 20)
        c = 50.0 * np.exp(-k * t)
        term = terminal_slope(t, c)
        assert term.is_estimable
        assert term.lambda_z_per_h == pytest.approx(k, rel=1e-6)
        assert term.half_life_h == pytest.approx(math.log(2) / k, rel=1e-6)

    def test_terminal_slope_abstains_on_too_few_points(self):
        term = terminal_slope(np.array([0.0, 1.0]), np.array([5.0, 4.0]))
        assert not term.is_estimable
        assert term.reason

    def test_terminal_slope_abstains_on_rising_profile(self):
        t = np.linspace(0, 10, 10)
        term = terminal_slope(t, np.linspace(1, 10, 10))
        assert not term.is_estimable

    def test_nca_recovers_clearance_from_dose_and_auc(self):
        """CL = Dose / AUC-inf must recover the simulation's own clearance."""
        m = OneCompartmentIV()
        dose = 10.0
        t = np.concatenate([[0.0], np.logspace(-2, np.log10(120), 400)])
        res = simulate(m, IV, [DoseEvent(0.0, dose, "central_mg")], t)
        r = nca(t, res.concentration_mg_per_l)
        assert r.auc_inf_mg_h_per_l is not None
        assert dose / r.auc_inf_mg_h_per_l == pytest.approx(IV["CL"], rel=1e-4)

    def test_nca_flags_unreliable_extrapolation(self):
        """Truncating the profile early makes AUC-inf extrapolation-dominated."""
        m = OneCompartmentIV()
        t = np.linspace(0, 2.0, 20)  # far short of the ~7 h half-life
        res = simulate(m, IV, [DoseEvent(0.0, 10.0, "central_mg")], t)
        r = nca(t, res.concentration_mg_per_l)
        assert r.extrapolated_fraction is not None
        assert r.extrapolated_fraction > 0.20
        assert not r.auc_inf_is_reliable

    def test_nca_accepts_well_characterised_profile(self):
        m = OneCompartmentIV()
        t = np.concatenate([[0.0], np.logspace(-2, np.log10(150), 300)])
        res = simulate(m, IV, [DoseEvent(0.0, 10.0, "central_mg")], t)
        assert nca(t, res.concentration_mg_per_l).auc_inf_is_reliable

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            nca(np.array([0.0, 1.0]), np.array([1.0]))


class TestSamplingBiasIsReal:
    """Sparse early sampling biases NCA AUC upward on multiphasic profiles.

    This is a property of the trapezoid rule, not a solver defect, and it is
    why the platform's quadrature uses log-spaced grids. Keeping it as a test
    documents the size of the effect rather than hiding it.
    """

    def test_uniform_coarse_grid_overestimates_two_compartment_auc(self):
        m = TwoCompartmentIV()
        p = {"CL": 0.1, "V1": 3.0, "Q": 10.0, "V2": 50.0}
        dose = 10.0
        exact = m.analytical_auc_inf(dose, p)
        th = m.terminal_half_life_h(p)

        uniform = np.linspace(0, 25 * th, 2001)
        c_uniform = m.analytical_concentration(uniform, dose, p)
        auc_uniform = auc_linlog(uniform, c_uniform) + c_uniform[-1] * th / math.log(2)

        logspaced = np.concatenate([[0.0], np.logspace(-4, np.log10(25 * th), 2000)])
        c_log = m.analytical_concentration(logspaced, dose, p)
        auc_log = auc_linlog(logspaced, c_log) + c_log[-1] * th / math.log(2)

        err_uniform = abs(auc_uniform - exact) / exact
        err_log = abs(auc_log - exact) / exact
        assert auc_uniform > exact, "coarse uniform sampling should overestimate"
        assert err_log < err_uniform / 10, (
            f"log-spaced grid should be far more accurate: "
            f"{err_log:.2e} vs {err_uniform:.2e}"
        )


class TestVerificationSuite:
    def test_representative_cases_pass_every_check(self):
        for name, cases in REFERENCE_CASES.items():
            model = MODEL_REGISTRY[name]()
            report = verify_model(model, cases[0])
            assert report.passed, str(report)
            assert len(report.checks) == 7

    def test_verification_detects_a_broken_model(self):
        """The suite must fail a model that does not conserve mass."""
        report = verify_model(LeakyModel(), IV)
        assert not report.passed
        assert any("mass_balance" in c.name for c in report.failures())
