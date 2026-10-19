"""Validate the mechanistic PK engine against observed human data.

Until now the solver was *verified*: it solved its equations correctly against
closed-form solutions. Verification says nothing about whether those equations
describe an organism. This module performs the other half, where the data
permit it.

The opportunity comes from over-determined labelling. Bremelanotide's FDA label
reports clearance, volume of distribution and bioavailability, and *separately*
reports the Cmax, AUC and terminal half-life observed in the same programme.
The first three parameterise the model; the second three are independent
observations the model can be asked to reproduce. Agreement is then a real test
rather than an identity.

Parameter uncertainty is propagated rather than ignored. The label states
CL = 6.5 ± 1.0 L/h and V = 25.0 ± 5.8 L, so the prediction is an interval, and
the question is whether the observed value falls inside it -- not whether a
point estimate happens to land close.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from atlas.core.evidence import utc_now
from atlas.core.units import convert, quantity
from atlas.mechanistic.metrics import nca
from atlas.mechanistic.models import OneCompartmentOral
from atlas.mechanistic.pk_extraction import LabelPKRecord, PKParameter, Qualifier
from atlas.mechanistic.solver import DoseEvent, SolverSettings, simulate

logger = logging.getLogger(__name__)

__all__ = [
    "ParameterSet",
    "ValidationTarget",
    "PKValidationReport",
    "build_parameter_set",
    "validate_against_label",
    "solve_ka_from_tmax",
]

#: Agreement within this relative error counts as reproducing the observation.
#: Twofold is the conventional bar for predictive PK; this is far stricter
#: because the parameters come from the same programme as the observations.
AGREEMENT_TOLERANCE = 0.20


def solve_ka_from_tmax(tmax_h: float, ke_per_h: float) -> float:
    """Absorption rate constant implied by an observed Tmax.

    ``tmax = ln(ka/ke)/(ka - ke)`` has no closed-form inverse, so it is solved
    by bisection. Tmax decreases monotonically in ka, which makes the bracket
    reliable.
    """
    if tmax_h <= 0 or ke_per_h <= 0:
        raise ValueError("tmax and ke must be positive")
    lo, hi = ke_per_h * (1 + 1e-9), ke_per_h * 1e6
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if mid <= ke_per_h * (1 + 1e-12):
            break
        t = math.log(mid / ke_per_h) / (mid - ke_per_h)
        if t > tmax_h:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class ParameterSet:
    """Model parameters assembled from label-extracted values."""

    model_name: str
    parameters: dict[str, float]
    dose_mg: float
    route: str | None
    derived: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    blocked_reason: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.blocked_reason is None


def _to(p: PKParameter, unit: str) -> float:
    return convert(quantity(p.value, p.unit), unit).value


def build_parameter_set(
    record: LabelPKRecord, *, body_weight_kg: float = 70.0
) -> ParameterSet:
    """Assemble a one-compartment first-order-absorption parameter set.

    Returns a blocked set, with the reason, when the label does not report
    enough to parameterise the model without inventing a value.
    """
    cl = record.get("clearance")
    vd = record.get("volume_of_distribution")
    f = record.get("bioavailability")
    tmax = record.get("tmax")

    missing = [n for n, p in
               (("clearance", cl), ("volume_of_distribution", vd),
                ("bioavailability", f), ("tmax", tmax)) if p is None]
    if missing:
        return ParameterSet(
            OneCompartmentOral.name, {}, record.dose_mg or 0.0, record.route,
            blocked_reason=(
                f"The label does not report {', '.join(missing)}, so the model "
                "cannot be parameterised without supplying a value the source "
                "does not contain."
            ),
        )
    if record.dose_mg is None:
        return ParameterSet(
            OneCompartmentOral.name, {}, 0.0, record.route,
            blocked_reason="No dose could be read from the dosage section.",
        )
    if not f.is_usable_as_point:
        return ParameterSet(
            OneCompartmentOral.name, {}, record.dose_mg, record.route,
            blocked_reason=(
                f"Bioavailability is reported as a bound ('{f.verbatim[:70]}...'), "
                "not a point estimate. Using the bound as a value would assert "
                "something the label does not."
            ),
        )

    cl_lh = _to(cl, "L/h")
    vd_l = _to(vd, "L")
    provenance = {
        "clearance": cl.verbatim,
        "volume_of_distribution": vd.verbatim,
        "bioavailability": f.verbatim,
        "tmax": tmax.verbatim,
    }
    derived: dict[str, str] = {}
    if vd.normalised_by_weight:
        vd_l = _to(vd, "L/kg") * body_weight_kg
        derived["volume_of_distribution"] = (
            f"Label reports {vd.value:g} {vd.unit}; multiplied by an assumed "
            f"{body_weight_kg:g} kg body weight, which the label does not state."
        )

    ke = cl_lh / vd_l
    ka = solve_ka_from_tmax(_to(tmax, "h"), ke)
    derived["ka"] = (
        f"Absorption rate constant {ka:.4g} /h solved from the reported "
        f"Tmax of {_to(tmax, 'h'):g} h and ke = CL/V = {ke:.4g} /h. The label "
        "does not report ka directly."
    )

    return ParameterSet(
        model_name=OneCompartmentOral.name,
        parameters={"CL": cl_lh, "V": vd_l, "ka": ka, "F": f.value},
        dose_mg=record.dose_mg, route=record.route,
        derived=derived, provenance=provenance,
    )


@dataclass
class ValidationTarget:
    """One independently reported quantity the model is asked to reproduce."""

    name: str
    observed: float
    unit: str
    predicted: float
    predicted_low: float | None = None
    predicted_high: float | None = None
    source_quote: str = ""

    @property
    def relative_error(self) -> float:
        return abs(self.predicted - self.observed) / abs(self.observed)

    @property
    def agrees(self) -> bool:
        return self.relative_error <= AGREEMENT_TOLERANCE

    @property
    def observed_within_interval(self) -> bool | None:
        if self.predicted_low is None or self.predicted_high is None:
            return None
        return self.predicted_low <= self.observed <= self.predicted_high

    def summary(self) -> str:
        interval = (
            f" [{self.predicted_low:.3g}, {self.predicted_high:.3g}]"
            if self.predicted_low is not None else ""
        )
        mark = "agrees" if self.agrees else "DIFFERS"
        return (
            f"{self.name}: predicted {self.predicted:.4g}{interval} vs observed "
            f"{self.observed:.4g} {self.unit} — {self.relative_error:.1%} "
            f"({mark})"
        )


@dataclass
class PKValidationReport:
    compound: str
    model_name: str
    parameter_set: ParameterSet
    targets: list[ValidationTarget] = field(default_factory=list)
    n_monte_carlo: int = 0
    limitations: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: utc_now().isoformat())

    @property
    def n_agreeing(self) -> int:
        return sum(1 for t in self.targets if t.agrees)

    @property
    def all_agree(self) -> bool:
        return bool(self.targets) and all(t.agrees for t in self.targets)

    @property
    def verdict(self) -> str:
        if not self.targets:
            return "no independent quantity available to validate against"
        if self.all_agree:
            return (
                f"reproduces all {len(self.targets)} independently reported "
                "quantities within tolerance"
            )
        agree = [t.name for t in self.targets if t.agrees]
        differ = [t.name for t in self.targets if not t.agrees]
        return (
            f"reproduces {', '.join(agree)} within tolerance; differs on "
            f"{', '.join(differ)}"
        )


def _predict(params: dict[str, float], dose_mg: float) -> dict[str, float]:
    """Run the real solver and return the derived quantities."""
    model = OneCompartmentOral()
    half_life = model.terminal_half_life_h(params)
    times = np.concatenate([[0.0], np.logspace(-3, np.log10(25 * half_life), 3000)])
    res = simulate(model, params, [DoseEvent(0.0, dose_mg, "depot_mg")], times,
                   settings=SolverSettings(rtol=1e-10, atol=1e-13))
    conc = res.concentration_mg_per_l
    analysis = nca(res.times_h, conc)
    tmax_h = model.analytical_tmax_h(params)
    return {
        "cmax": float(np.max(conc)) * 1000.0,          # mg/L -> ng/mL
        "auc": (model.analytical_auc_inf(dose_mg, params) or 0.0) * 1000.0,
        "half_life": half_life,
        "tmax": tmax_h,
        "auc_numeric": analysis.auc_inf_mg_h_per_l * 1000.0
        if analysis.auc_inf_mg_h_per_l else float("nan"),
    }


def validate_against_label(
    record: LabelPKRecord, *, n_monte_carlo: int = 2000, seed: int = 20261018,
    body_weight_kg: float = 70.0,
) -> PKValidationReport:
    """Reproduce the label's independently reported quantities, with uncertainty."""
    ps = build_parameter_set(record, body_weight_kg=body_weight_kg)
    report = PKValidationReport(
        compound=record.generic_name, model_name=ps.model_name, parameter_set=ps,
        n_monte_carlo=n_monte_carlo,
    )
    if not ps.is_usable:
        report.limitations.append(ps.blocked_reason or "parameter set unusable")
        return report

    point = _predict(ps.parameters, ps.dose_mg)

    # Propagate the spread the label itself reports.
    cl_p = record.get("clearance")
    vd_p = record.get("volume_of_distribution")
    tmax_p = record.get("tmax")
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"cmax": [], "auc": [], "half_life": []}

    cl_sd = _to_sd(cl_p, "L/h")
    vd_sd = _to_sd(vd_p, "L") if not vd_p.normalised_by_weight else None
    for _ in range(n_monte_carlo):
        cl = ps.parameters["CL"] if cl_sd is None else max(
            1e-6, rng.normal(ps.parameters["CL"], cl_sd))
        v = ps.parameters["V"] if vd_sd is None else max(
            1e-6, rng.normal(ps.parameters["V"], vd_sd))
        if tmax_p.range_low and tmax_p.range_high:
            tmax = rng.uniform(tmax_p.range_low, tmax_p.range_high)
        else:
            tmax = _to(tmax_p, "h")
        ke = cl / v
        try:
            ka = solve_ka_from_tmax(tmax, ke)
        except ValueError:
            continue
        p = {"CL": cl, "V": v, "ka": ka, "F": ps.parameters["F"]}
        out = _predict(p, ps.dose_mg)
        for k in draws:
            draws[k].append(out[k])

    def interval(key: str) -> tuple[float | None, float | None]:
        vals = draws.get(key) or []
        if len(vals) < 50:
            return None, None
        return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))

    # Compare against quantities the label reports independently.
    for name, unit in (("cmax", "ng/mL"), ("auc", "ng*h/mL"), ("half_life", "h")):
        observed = record.get(name)
        if observed is None:
            continue
        obs_value = (
            _to(observed, unit) if name != "half_life" else _to(observed, "h")
        )
        lo, hi = interval(name)
        report.targets.append(ValidationTarget(
            name=name, observed=obs_value, unit=unit,
            predicted=point[name], predicted_low=lo, predicted_high=hi,
            source_quote=observed.verbatim,
        ))

    report.limitations.extend([
        "Parameters and the quantities they are compared against come from the "
        "same labelling document, so this is internal consistency plus "
        "structural adequacy, not independent external validation.",
        "A one-compartment model with first-order absorption is the simplest "
        "structure that fits the reported parameters. Agreement does not "
        "establish that the true disposition is one-compartment.",
        "The label reports population means; no individual-level data are "
        "available, so between-subject variability is represented only by the "
        "standard deviations the label states.",
    ])
    if ps.derived:
        report.limitations.append(
            "Some parameters were derived rather than read: "
            + "; ".join(ps.derived.values())
        )
    return report


def _to_sd(p: PKParameter | None, unit: str) -> float | None:
    if p is None or p.sd is None:
        return None
    return convert(quantity(p.sd, p.unit), unit).value
