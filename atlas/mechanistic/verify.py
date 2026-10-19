"""Solver verification: does the code solve the equations it claims to solve?

This is verification, not validation, and the distinction is load-bearing.
Verification compares numerical output against exact mathematics -- closed-form
solutions, conservation laws and analytic identities -- and can be settled to
machine precision without any experimental data. Validation asks whether those
equations describe a real biological system, which requires observed
measurements and cannot be established here.

A model that passes every check below is correctly implemented and still says
nothing about any compound's behaviour in an organism.

The identities exercised are exact for linear systems:

* ``AUC(0, inf) == F * Dose / CL``
* ``eliminated(inf) == F * Dose``
* ``C(k * Dose) == k * C(Dose)`` (superposition)
* ``mean steady-state concentration == F * Dose / (CL * tau)``
* total drug across all states equals the dose administered so far
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from atlas.mechanistic.metrics import auc_linlog
from atlas.mechanistic.models import (
    MODEL_REGISTRY,
    OneCompartmentIV,
    OneCompartmentOral,
    PKModel,
    TwoCompartmentIV,
)
from atlas.mechanistic.solver import DoseEvent, SolverSettings, simulate

__all__ = ["Check", "VerificationReport", "verify_model", "verify_all", "REFERENCE_CASES"]


@dataclass(frozen=True, slots=True)
class Check:
    """One verification check with the quantity it measured."""

    name: str
    passed: bool
    metric: float
    tolerance: float
    units: str
    detail: str

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.name}: {self.metric:.3e} {self.units} "
            f"(tolerance {self.tolerance:.1e}) -- {self.detail}"
        )


@dataclass
class VerificationReport:
    model_name: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def __str__(self) -> str:
        head = f"{self.model_name}: {self.n_passed}/{len(self.checks)} checks passed"
        return "\n".join([head, *(f"  {c}" for c in self.checks)])


def _dose_compartment(model: PKModel) -> str:
    """Depot for extravascular models, central compartment otherwise."""
    return "depot_mg" if "depot_mg" in model.state_names else "central_mg"


def _grid(model: PKModel, p: Mapping[str, float], *, n: int = 400, spans: float = 12.0):
    """Uniform time grid spanning several terminal half-lives."""
    thalf = model.terminal_half_life_h(p) or 1.0
    return np.linspace(0.0, spans * thalf, n)


def _fastest_rate_per_h(model: PKModel, p: Mapping[str, float]) -> float:
    """Rate constant of the quickest process, used to size early sampling."""
    rates = [p["CL"] / p.get("V", p.get("V1", 1.0))]
    if "ka" in p:
        rates.append(p["ka"])
    if isinstance(model, TwoCompartmentIV):
        alpha, _ = model._hybrid(p)
        rates.append(alpha)
    return max(r for r in rates if r > 0)


def _quadrature_grid(
    model: PKModel, p: Mapping[str, float], *, n: int = 4000, spans: float = 25.0
) -> np.ndarray:
    """Phase-aware grid that resolves fast and slow phases alike.

    A uniform grid spanning many terminal half-lives barely samples a rapid
    distribution phase, and the trapezoid rule then overestimates the steep
    early decline. Real PK sampling schedules are log-spaced for the same
    reason, so quadrature checks use a log-spaced grid.
    """
    thalf = model.terminal_half_life_h(p) or 1.0
    t_end = spans * thalf
    t_min = min(1.0 / _fastest_rate_per_h(model, p), thalf) / 1000.0
    return np.concatenate(
        [[0.0], np.logspace(np.log10(t_min), np.log10(t_end), n)]
    )


def check_analytical_agreement(
    model: PKModel, p: Mapping[str, float], dose_mg: float, *, tol: float = 1e-6
) -> Check:
    """Numerical trajectory against the closed-form solution."""
    times = _grid(model, p)
    exact = model.analytical_concentration(times, dose_mg, p)
    if exact is None:
        return Check(
            "analytical_agreement", True, 0.0, tol, "relative",
            "no closed form implemented; check skipped",
        )
    res = simulate(
        model, p, [DoseEvent(0.0, dose_mg, _dose_compartment(model))], times,
        settings=SolverSettings(rtol=1e-11, atol=1e-14),
    )
    num = res.concentration_mg_per_l
    scale = float(np.max(np.abs(exact)))
    err = float(np.max(np.abs(num - exact))) / max(scale, 1e-30)
    return Check(
        "analytical_agreement", err <= tol, err, tol, "relative",
        f"max normalised deviation from the exact solution over {times.size} points",
    )


def check_mass_balance(
    model: PKModel, p: Mapping[str, float], dose_mg: float, *, tol: float = 1e-9
) -> Check:
    times = _grid(model, p)
    res = simulate(
        model, p, [DoseEvent(0.0, dose_mg, _dose_compartment(model))], times,
        settings=SolverSettings(rtol=1e-11, atol=1e-14),
    )
    err = res.mass_balance_relative_error
    return Check(
        "mass_balance", err <= tol, err, tol, "relative",
        "total drug across all states against cumulative dose",
    )


def check_auc_identity(
    model: PKModel, p: Mapping[str, float], dose_mg: float, *, tol: float = 1e-4
) -> Check:
    """Numerically integrated AUC against the exact ``F * Dose / CL``."""
    exact = model.analytical_auc_inf(dose_mg, p)
    if exact is None:
        return Check("auc_identity", True, 0.0, tol, "relative",
                     "no analytic AUC available; check skipped")
    # Integrate far enough that the untracked tail is negligible, then add it.
    # The grid must resolve every phase, not merely span the terminal one.
    times = _quadrature_grid(model, p)
    res = simulate(
        model, p, [DoseEvent(0.0, dose_mg, _dose_compartment(model))], times,
        settings=SolverSettings(rtol=1e-11, atol=1e-14),
    )
    conc = res.concentration_mg_per_l
    auc = auc_linlog(times, conc)
    lam = np.log(2.0) / (model.terminal_half_life_h(p) or 1.0)
    auc += float(conc[-1]) / lam  # analytic mono-exponential tail
    err = abs(auc - exact) / abs(exact)
    return Check(
        "auc_identity", err <= tol, err, tol, "relative",
        f"integrated AUC {auc:.6g} against exact F*Dose/CL = {exact:.6g} mg*h/L",
    )


def check_eliminated_fraction(
    model: PKModel, p: Mapping[str, float], dose_mg: float, *, tol: float = 1e-5
) -> Check:
    """At long times the eliminated amount must equal the bioavailable dose."""
    if "eliminated_mg" not in model.state_names:
        return Check("eliminated_fraction", True, 0.0, tol, "relative",
                     "model has no elimination state; check skipped")
    f = float(p.get("F", 1.0))
    times = _grid(model, p, n=2000, spans=40.0)
    res = simulate(
        model, p, [DoseEvent(0.0, dose_mg, _dose_compartment(model))], times,
        settings=SolverSettings(rtol=1e-11, atol=1e-14),
    )
    eliminated = float(res.state("eliminated_mg")[-1])
    expected = f * dose_mg
    err = abs(eliminated - expected) / max(expected, 1e-30)
    return Check(
        "eliminated_fraction", err <= tol, err, tol, "relative",
        f"eliminated {eliminated:.6g} mg against bioavailable dose "
        f"{expected:.6g} mg after 40 half-lives",
    )


def check_dose_proportionality(
    model: PKModel, p: Mapping[str, float], dose_mg: float, *, tol: float = 1e-8
) -> Check:
    """Superposition: these systems are linear, so scaling dose scales output."""
    times = _grid(model, p)
    comp = _dose_compartment(model)
    settings = SolverSettings(rtol=1e-11, atol=1e-14)
    low = simulate(model, p, [DoseEvent(0.0, dose_mg, comp)], times, settings=settings)
    high = simulate(
        model, p, [DoseEvent(0.0, 3.0 * dose_mg, comp)], times, settings=settings
    )
    scaled = 3.0 * low.concentration_mg_per_l
    scale = float(np.max(np.abs(scaled)))
    err = float(np.max(np.abs(high.concentration_mg_per_l - scaled))) / max(scale, 1e-30)
    return Check(
        "dose_proportionality", err <= tol, err, tol, "relative",
        "tripling the dose against tripling the concentration trajectory",
    )


def check_steady_state_identity(
    model: PKModel, p: Mapping[str, float], dose_mg: float, *, tol: float = 1e-4
) -> Check:
    r"""Average steady-state concentration must equal :math:`F D /(CL \tau)`."""
    thalf = model.terminal_half_life_h(p) or 1.0
    tau = max(thalf / 2.0, 1e-3)
    n_doses = 40  # ~20 half-lives of accumulation
    comp = _dose_compartment(model)
    doses = [DoseEvent(i * tau, dose_mg, comp) for i in range(n_doses)]

    # Sample the final dosing interval densely.
    start = (n_doses - 1) * tau
    times = np.concatenate(
        [np.array([0.0]), np.linspace(start, start + tau, 4001)]
    )
    res = simulate(
        model, p, doses, times, settings=SolverSettings(rtol=1e-11, atol=1e-14)
    )
    interval = res.concentration_mg_per_l[1:]
    interval_times = times[1:]
    mean_obs = float(np.trapezoid(interval, interval_times)) / tau if hasattr(
        np, "trapezoid"
    ) else float(np.trapz(interval, interval_times)) / tau
    f = float(p.get("F", 1.0))
    mean_exact = f * dose_mg / (p["CL"] * tau)
    err = abs(mean_obs - mean_exact) / abs(mean_exact)
    return Check(
        "steady_state_identity", err <= tol, err, tol, "relative",
        f"mean C over the 40th interval {mean_obs:.6g} against "
        f"F*Dose/(CL*tau) = {mean_exact:.6g} mg/L",
    )


def check_solver_convergence(
    model: PKModel, p: Mapping[str, float], dose_mg: float
) -> Check:
    """Tightening tolerance must reduce the error against the exact solution."""
    times = _grid(model, p, n=200)
    exact = model.analytical_concentration(times, dose_mg, p)
    if exact is None:
        return Check("solver_convergence", True, 0.0, 1.0, "ratio",
                     "no closed form implemented; check skipped")
    comp = _dose_compartment(model)
    scale = max(float(np.max(np.abs(exact))), 1e-30)
    errors = []
    for rtol in (1e-5, 1e-8, 1e-11):
        res = simulate(
            model, p, [DoseEvent(0.0, dose_mg, comp)], times,
            settings=SolverSettings(rtol=rtol, atol=rtol * 1e-3),
        )
        errors.append(
            float(np.max(np.abs(res.concentration_mg_per_l - exact))) / scale
        )
    # Monotone improvement, allowing for a floor at machine precision.
    improved = errors[-1] <= errors[0] and errors[-1] < 1e-7
    ratio = errors[-1] / max(errors[0], 1e-30)
    return Check(
        "solver_convergence", improved, ratio, 1.0, "ratio",
        f"error at rtol 1e-5 -> 1e-11: {errors[0]:.2e} -> {errors[-1]:.2e}",
    )


#: Parameter sets used for verification. These are *mathematical* test cases
#: spanning qualitatively different regimes (fast and slow absorption,
#: flip-flop kinetics, distribution-dominated disposition). They are not
#: claims about any compound and are deliberately not attributed to one.
REFERENCE_CASES: dict[str, list[dict[str, float]]] = {
    OneCompartmentIV.name: [
        {"CL": 1.0, "V": 10.0},
        {"CL": 0.05, "V": 12.5},
        {"CL": 25.0, "V": 3.0},
    ],
    OneCompartmentOral.name: [
        {"CL": 1.0, "V": 10.0, "ka": 1.5, "F": 0.8},
        {"CL": 0.05, "V": 12.5, "ka": 0.03, "F": 0.9},   # slow absorption
        {"CL": 2.0, "V": 10.0, "ka": 0.05, "F": 1.0},    # flip-flop regime
        {"CL": 1.0, "V": 10.0, "ka": 0.1, "F": 1.0},     # ka == ke singularity
    ],
    TwoCompartmentIV.name: [
        {"CL": 1.0, "V1": 5.0, "Q": 2.0, "V2": 20.0},
        {"CL": 0.1, "V1": 3.0, "Q": 10.0, "V2": 50.0},
    ],
}


def verify_model(
    model: PKModel, params: Mapping[str, float], *, dose_mg: float = 10.0
) -> VerificationReport:
    """Run every applicable check for one model and parameter set."""
    report = VerificationReport(model_name=f"{model.name} {dict(params)}")
    for fn in (
        check_analytical_agreement,
        check_mass_balance,
        check_auc_identity,
        check_eliminated_fraction,
        check_dose_proportionality,
        check_steady_state_identity,
        check_solver_convergence,
    ):
        try:
            report.checks.append(fn(model, params, dose_mg))
        except Exception as exc:  # a raising check is a failing check
            report.checks.append(
                Check(fn.__name__.removeprefix("check_"), False, float("nan"),
                      0.0, "", f"check raised: {type(exc).__name__}: {exc}")
            )
    return report


def verify_all() -> list[VerificationReport]:
    """Verify every registered model across all reference parameter sets."""
    reports: list[VerificationReport] = []
    for name, cases in REFERENCE_CASES.items():
        model = MODEL_REGISTRY[name]()
        for params in cases:
            reports.append(verify_model(model, params))
    return reports


def main() -> int:
    reports = verify_all()
    for r in reports:
        print(r)
        print()
    failed = [r for r in reports if not r.passed]
    total_checks = sum(len(r.checks) for r in reports)
    passed_checks = sum(r.n_passed for r in reports)
    print(f"{'=' * 72}")
    print(
        f"VERIFICATION: {passed_checks}/{total_checks} checks passed across "
        f"{len(reports)} model/parameter combinations"
    )
    if failed:
        print(f"FAILING COMBINATIONS: {len(failed)}")
        return 1
    print("All solver verification checks passed.")
    print(
        "Note: this establishes numerical correctness only. It is not evidence "
        "that any model describes a real biological system."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
