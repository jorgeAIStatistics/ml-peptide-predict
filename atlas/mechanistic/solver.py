"""Numerical integration of compartmental models with run provenance.

Repeated dosing is handled by segmenting the integration at each dose event
rather than by adding a forcing term, because a bolus is a discontinuity in the
state and smooth integrators silently lose accuracy across it.

Every run returns solver settings, a mass-balance diagnostic and a run id, so a
stored trajectory can be reproduced and audited.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from atlas.core.evidence import utc_now
from atlas.mechanistic.models import PKModel, ParameterError

__all__ = [
    "DoseEvent",
    "SimulationResult",
    "SolverSettings",
    "SolverError",
    "simulate",
]


class SolverError(RuntimeError):
    """Raised when numerical integration fails or violates conservation."""


@dataclass(frozen=True, slots=True)
class DoseEvent:
    """A dose administered at ``time_h`` into ``compartment``.

    ``compartment`` is a state name: the depot for extravascular administration,
    the central compartment for intravenous.
    """

    time_h: float
    amount_mg: float
    compartment: str

    def __post_init__(self) -> None:
        if self.time_h < 0:
            raise ValueError("dose time must be non-negative")
        if self.amount_mg <= 0:
            raise ValueError("dose amount must be positive")


@dataclass(frozen=True, slots=True)
class SolverSettings:
    method: Literal["LSODA", "Radau", "BDF", "RK45", "DOP853"] = "LSODA"
    rtol: float = 1e-9
    atol: float = 1e-12
    max_step: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step": self.max_step,
        }


@dataclass(frozen=True)
class SimulationResult:
    """Trajectories plus everything needed to reproduce and audit them."""

    model_name: str
    parameters: dict[str, float]
    doses: tuple[DoseEvent, ...]
    times_h: np.ndarray
    states_mg: np.ndarray  # shape (n_states, n_times)
    concentration_mg_per_l: np.ndarray
    state_names: tuple[str, ...]
    settings: SolverSettings
    mass_balance_max_abs_error_mg: float
    total_dose_mg: float
    created_at: str = field(default_factory=lambda: utc_now().isoformat())

    @property
    def run_id(self) -> str:
        payload = {
            "model": self.model_name,
            "parameters": self.parameters,
            "doses": [(d.time_h, d.amount_mg, d.compartment) for d in self.doses],
            "times": [float(self.times_h[0]), float(self.times_h[-1]), len(self.times_h)],
            "settings": self.settings.as_dict(),
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def mass_balance_relative_error(self) -> float:
        return self.mass_balance_max_abs_error_mg / max(self.total_dose_mg, 1e-30)

    def concentration_ng_per_ml(self) -> np.ndarray:
        """mg/L and ug/mL are the same unit; ng/mL is 1000x that."""
        return self.concentration_mg_per_l * 1000.0

    def state(self, name: str) -> np.ndarray:
        try:
            return self.states_mg[self.state_names.index(name)]
        except ValueError:
            raise KeyError(
                f"no state {name!r}; model has {list(self.state_names)}"
            ) from None


def _dose_index(model: PKModel, compartment: str) -> int:
    try:
        return model.state_names.index(compartment)
    except ValueError:
        raise ParameterError(
            f"model {model.name} has no compartment {compartment!r}; "
            f"available: {list(model.state_names)}"
        ) from None


def simulate(
    model: PKModel,
    params: Mapping[str, float],
    doses: Sequence[DoseEvent],
    times_h: Sequence[float] | np.ndarray,
    *,
    settings: SolverSettings | None = None,
    mass_balance_rtol: float = 1e-6,
) -> SimulationResult:
    """Integrate ``model`` over ``times_h`` under a dosing schedule.

    Raises :class:`SolverError` if integration fails or if mass balance drifts
    beyond ``mass_balance_rtol``, which would mean the trajectory is not a
    solution of the stated system.
    """
    if not doses:
        raise ValueError("at least one dose event is required")

    settings = settings or SolverSettings()
    p = model.validate(params)
    times = np.asarray(times_h, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times_h must be a non-empty 1-D sequence")
    if np.any(np.diff(times) < 0):
        raise ValueError("times_h must be non-decreasing")
    if times[0] < 0:
        raise ValueError("times_h must be non-negative")

    ordered = sorted(doses, key=lambda d: d.time_h)
    if ordered[0].time_h > times[0]:
        raise ValueError(
            "the first dose must occur at or before the first output time"
        )

    n_states = len(model.state_names)
    y = np.zeros(n_states)
    out = np.full((n_states, times.size), np.nan)

    # Integration proceeds segment by segment between dose events. Boundaries
    # are the dose times plus the final output time.
    boundaries: list[float] = []
    for d in ordered:
        if d.time_h not in boundaries:
            boundaries.append(d.time_h)
    end = float(times[-1])
    if end > boundaries[-1]:
        boundaries.append(end)
    elif len(boundaries) == 1:
        boundaries.append(max(end, boundaries[0] + 1e-9))

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        return model.rhs(t, state, p)

    # Every boundary is visited, including the last: a dose scheduled exactly at
    # the final output time must still be administered and recorded.
    for i, seg_start in enumerate(boundaries):
        # Apply every dose scheduled exactly at this boundary.
        for d in ordered:
            if d.time_h == seg_start:
                y = y.copy()
                y[_dose_index(model, d.compartment)] += d.amount_mg

        # Record output times coinciding with the boundary. This runs after
        # dosing and after the preceding segment wrote its endpoint, so a
        # sample taken at a dose time reports the post-dose state.
        at_boundary = np.isclose(times, seg_start, rtol=0, atol=1e-12)
        if np.any(at_boundary):
            out[:, at_boundary] = y[:, None]

        if i == len(boundaries) - 1:
            break
        seg_end = boundaries[i + 1]

        inside = (times > seg_start) & (times <= seg_end)
        eval_times = times[inside]
        if seg_end <= seg_start:
            continue

        sol = solve_ivp(
            rhs,
            (seg_start, seg_end),
            y,
            method=settings.method,
            rtol=settings.rtol,
            atol=settings.atol,
            max_step=settings.max_step if settings.max_step else np.inf,
            t_eval=eval_times if eval_times.size else None,
            dense_output=True,
        )
        if not sol.success:
            raise SolverError(
                f"integration failed on [{seg_start:g}, {seg_end:g}] h "
                f"for model {model.name}: {sol.message}"
            )
        if eval_times.size:
            out[:, inside] = sol.y
        y = sol.sol(seg_end)

    if np.isnan(out).any():
        missing = times[np.isnan(out).any(axis=0)]
        raise SolverError(f"no solution produced at times {missing[:5]}")

    # Mass balance: the sum of all states must equal the dose administered so
    # far at every output time.
    cumulative_dose = np.array(
        [sum(d.amount_mg for d in ordered if d.time_h <= t) for t in times]
    )
    totals = out.sum(axis=0)
    abs_err = float(np.max(np.abs(totals - cumulative_dose)))
    total_dose = float(sum(d.amount_mg for d in ordered))
    if abs_err > mass_balance_rtol * max(total_dose, 1e-30):
        raise SolverError(
            f"mass balance violated for {model.name}: maximum absolute error "
            f"{abs_err:.3e} mg against a total dose of {total_dose:g} mg "
            f"(tolerance {mass_balance_rtol:g} relative)"
        )

    conc = model.concentration(out[model.central_index], p)

    return SimulationResult(
        model_name=model.name,
        parameters=p,
        doses=tuple(ordered),
        times_h=times,
        states_mg=out,
        concentration_mg_per_l=conc,
        state_names=tuple(model.state_names),
        settings=settings,
        mass_balance_max_abs_error_mg=abs_err,
        total_dose_mg=total_dose,
    )
