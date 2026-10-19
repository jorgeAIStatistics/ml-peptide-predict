"""Non-compartmental pharmacokinetic metrics computed from trajectories.

These are the quantities reported in PK papers (AUC, Cmax, Tmax, terminal
half-life), computed the way pharmacometricians compute them, including the
linear-up/log-down trapezoid rule that is standard for the decline phase.

Each metric reports how it was derived, because an AUC to the last observed
time and an AUC extrapolated to infinity are different quantities and the
extrapolated fraction determines whether the latter is trustworthy at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "auc_linear",
    "auc_linlog",
    "cmax_tmax",
    "terminal_slope",
    "TerminalPhase",
    "NCAResult",
    "nca",
]


def auc_linear(t: np.ndarray, c: np.ndarray) -> float:
    """Linear trapezoidal AUC."""
    t = np.asarray(t, float)
    c = np.asarray(c, float)
    if t.size < 2:
        return 0.0
    return float(np.trapezoid(c, t)) if hasattr(np, "trapezoid") else float(
        np.trapz(c, t)
    )


def auc_linlog(t: np.ndarray, c: np.ndarray) -> float:
    """Linear-up / log-down trapezoidal AUC.

    Uses the logarithmic trapezoid on intervals where concentration falls and
    both endpoints are positive, which is exact for mono-exponential decline
    and avoids the systematic overestimate of the linear rule.
    """
    t = np.asarray(t, float)
    c = np.asarray(c, float)
    if t.size < 2:
        return 0.0
    total = 0.0
    for i in range(t.size - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0:
            continue
        c0, c1 = c[i], c[i + 1]
        if c1 < c0 and c0 > 0 and c1 > 0:
            # Exact integral of an exponential through the two endpoints.
            total += dt * (c0 - c1) / math.log(c0 / c1)
        else:
            total += dt * 0.5 * (c0 + c1)
    return float(total)


def cmax_tmax(t: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """Observed maximum concentration and the time at which it occurs.

    This is the *observed* maximum over the sampled grid, which is generally
    below the true maximum unless sampling is dense near the peak.
    """
    c = np.asarray(c, float)
    idx = int(np.argmax(c))
    return float(c[idx]), float(np.asarray(t, float)[idx])


@dataclass(frozen=True, slots=True)
class TerminalPhase:
    """Result of fitting the terminal log-linear phase."""

    lambda_z_per_h: float | None
    half_life_h: float | None
    n_points: int
    r_squared: float | None
    first_time_h: float | None
    reason: str | None = None

    @property
    def is_estimable(self) -> bool:
        return self.lambda_z_per_h is not None


def terminal_slope(
    t: np.ndarray, c: np.ndarray, *, min_points: int = 3
) -> TerminalPhase:
    """Estimate the terminal elimination rate constant by log-linear regression.

    Chooses the window maximising adjusted R-squared over the final points,
    which is the conventional automated rule. Returns a non-estimable result
    with a reason rather than a number when the data cannot support a fit.
    """
    t = np.asarray(t, float)
    c = np.asarray(c, float)
    mask = c > 0
    t, c = t[mask], c[mask]
    if t.size < min_points:
        return TerminalPhase(None, None, t.size, None, None,
                             "fewer than three positive concentrations")

    peak = int(np.argmax(c))
    # The terminal phase can only begin after the peak.
    t_tail, c_tail = t[peak:], c[peak:]
    if t_tail.size < min_points:
        return TerminalPhase(None, None, t_tail.size, None, None,
                             "fewer than three points after the peak")

    best: TerminalPhase | None = None
    best_adj = -np.inf
    log_c = np.log(c_tail)

    for n in range(min_points, t_tail.size + 1):
        tt, ll = t_tail[-n:], log_c[-n:]
        slope, intercept = np.polyfit(tt, ll, 1)
        if slope >= 0:
            continue  # not a decline; cannot be a terminal elimination phase
        pred = slope * tt + intercept
        ss_res = float(np.sum((ll - pred) ** 2))
        ss_tot = float(np.sum((ll - ll.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - 2, 1)
        if adj > best_adj:
            best_adj = adj
            lam = -float(slope)
            best = TerminalPhase(
                lambda_z_per_h=lam,
                half_life_h=math.log(2.0) / lam,
                n_points=n,
                r_squared=r2,
                first_time_h=float(tt[0]),
            )

    if best is None:
        return TerminalPhase(None, None, t_tail.size, None, None,
                             "no declining window found after the peak")
    return best


@dataclass(frozen=True)
class NCAResult:
    """Non-compartmental analysis of one concentration-time profile."""

    cmax_mg_per_l: float
    tmax_h: float
    auc_last_mg_h_per_l: float
    auc_last_method: Literal["linear", "linlog"]
    terminal: TerminalPhase
    auc_inf_mg_h_per_l: float | None
    extrapolated_fraction: float | None
    t_last_h: float
    c_last_mg_per_l: float

    @property
    def auc_inf_is_reliable(self) -> bool:
        """Conventional acceptance: under 20% of AUC from extrapolation.

        Beyond that, AUC-infinity is driven by the assumed terminal slope
        rather than by observed data.
        """
        return (
            self.extrapolated_fraction is not None
            and self.extrapolated_fraction < 0.20
        )

    @property
    def clearance_l_per_h(self) -> float | None:
        """CL/F from dose and AUC-infinity. Requires the dose, supplied by caller."""
        return None


def nca(
    t: np.ndarray,
    c: np.ndarray,
    *,
    method: Literal["linear", "linlog"] = "linlog",
) -> NCAResult:
    """Run non-compartmental analysis on a concentration-time profile."""
    t = np.asarray(t, float)
    c = np.asarray(c, float)
    if t.size != c.size:
        raise ValueError("time and concentration must have the same length")
    if t.size < 2:
        raise ValueError("at least two samples are required")

    cmax, tmax = cmax_tmax(t, c)
    auc_last = auc_linlog(t, c) if method == "linlog" else auc_linear(t, c)
    term = terminal_slope(t, c)

    auc_inf: float | None = None
    extrap: float | None = None
    if term.is_estimable and c[-1] > 0:
        # Tail area of a mono-exponential beyond the last observation.
        tail = c[-1] / term.lambda_z_per_h
        auc_inf = auc_last + tail
        extrap = tail / auc_inf if auc_inf > 0 else None

    return NCAResult(
        cmax_mg_per_l=cmax,
        tmax_h=tmax,
        auc_last_mg_h_per_l=auc_last,
        auc_last_method=method,
        terminal=term,
        auc_inf_mg_h_per_l=auc_inf,
        extrapolated_fraction=extrap,
        t_last_h=float(t[-1]),
        c_last_mg_per_l=float(c[-1]),
    )
