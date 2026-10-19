"""Turn a number into something a researcher can act on.

Every interpretation here is *computed* from the value and from the metadata of
the model that produced it. None of it is prewritten narrative attached to a
result after the fact. A pKi of 5.45 is reported as a Ki of 3.55 uM because
10^-5.45 M was evaluated, and an interval of +/-1.21 log units is reported as a
16-fold range because 10^(2x1.21) was evaluated.

The structure follows the six questions a result has to answer: what was
measured, what the number means mathematically, what it implies biologically,
how uncertain it is, why the model produced it, and what would confirm or
refute it.

Where two quantities can be related, the relation is computed rather than left
to the reader. A peak concentration and a measured Ki together give a
concentration-to-affinity ratio, which is the bridge from exposure to target
engagement -- stated with the caveats that make it honest, since plasma is not
tissue and total is not free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from atlas.interpretation.definitions import EndpointDefinition, get_definition

__all__ = ["Interpretation", "interpret", "concentration_vs_affinity"]


@dataclass
class Interpretation:
    """One number, explained."""

    endpoint: str
    symbol: str
    name: str
    value: float | None
    unit: str

    what_is_measured: dict[str, Any] = field(default_factory=dict)
    mathematics: dict[str, Any] = field(default_factory=dict)
    derived: list[dict[str, Any]] = field(default_factory=list)
    biological_meaning: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    why_this_result: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint, "symbol": self.symbol, "name": self.name,
            "value": self.value, "unit": self.unit,
            "what_is_measured": self.what_is_measured,
            "mathematics": self.mathematics,
            "derived": self.derived,
            "biological_meaning": self.biological_meaning,
            "uncertainty": self.uncertainty,
            "why_this_result": self.why_this_result,
            "validation": self.validation,
        }


def _fmt(v: float, sig: int = 3) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.2e}"
    return f"{float(f'%.{sig}g' % v):g}"


def _concentration_phrase(molar: float) -> str:
    """Express a molar concentration in whichever unit reads naturally.

    The tolerance matters: 10**-6.0 evaluates to slightly below 1e-6 in IEEE
    arithmetic, so a strict comparison rendered a pKi of 6.0 as "1000 nM"
    instead of "1 uM". Units are chosen from the largest scale the value
    reaches, allowing for that representation error.
    """
    if molar <= 0:
        return "0"
    for scale, unit in ((1e-3, "mM"), (1e-6, "uM"), (1e-9, "nM"), (1e-12, "pM")):
        if molar >= scale * (1 - 1e-9):
            return f"{_fmt(molar / scale)} {unit}"
    return f"{_fmt(molar)} M"


def _derive(endpoint: str, value: float, ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Quantities that follow from this one by exact transformation."""
    out: list[dict[str, Any]] = []

    if endpoint == "pki":
        ki_molar = 10 ** (-value)
        out.append({
            "label": "Equivalent Ki",
            "value": _concentration_phrase(ki_molar),
            "how": f"Ki = 10^(−{_fmt(value)}) M",
        })
        out.append({
            "label": "Meaning of one unit",
            "value": "tenfold change in Ki",
            "how": "pKi is base-10 logarithmic, so +1 is a 10x tighter Ki",
        })
        lo, hi = ctx.get("interval_low"), ctx.get("interval_high")
        if lo is not None and hi is not None:
            fold = 10 ** (hi - lo)
            out.append({
                "label": "Interval expressed as affinity",
                "value": (
                    f"{_concentration_phrase(10 ** (-hi))} to "
                    f"{_concentration_phrase(10 ** (-lo))} "
                    f"({_fmt(fold)}-fold range)"
                ),
                "how": (
                    f"a {_fmt(hi - lo)} log-unit interval becomes "
                    f"10^{_fmt(hi - lo)} = {_fmt(fold)}-fold in Ki"
                ),
            })

    elif endpoint == "half_life":
        ke = math.log(2) / value if value > 0 else None
        if ke:
            out.append({
                "label": "Elimination rate constant",
                "value": f"{_fmt(ke)} /h",
                "how": f"ke = ln(2)/{_fmt(value)} h",
            })
            out.append({
                "label": "Time to approximate steady state",
                "value": f"{_fmt(4.3 * value)} h",
                "how": "about 4.3 half-lives reaches 95% of steady state",
            })
        tau = ctx.get("dosing_interval_h")
        if tau and value > 0:
            ratio = 1.0 / (1.0 - 2 ** (-tau / value))
            out.append({
                "label": f"Accumulation ratio at a {_fmt(tau)} h interval",
                "value": f"{_fmt(ratio)}x",
                "how": "R = 1/(1 − 2^(−tau/t½)) for a linear system",
            })

    elif endpoint == "auc":
        dose_mg = ctx.get("dose_mg")
        f = ctx.get("bioavailability")
        if dose_mg and value > 0:
            # AUC in ng*h/mL equals mg*h/L, so CL in L/h follows directly.
            cl = (f if f is not None else 1.0) * dose_mg / (value / 1000.0)
            out.append({
                "label": "Implied clearance",
                "value": f"{_fmt(cl)} L/h",
                "how": (
                    f"CL = F·Dose/AUC = {_fmt(f if f is not None else 1.0)}·"
                    f"{_fmt(dose_mg)} mg / {_fmt(value / 1000.0)} mg·h/L"
                    + ("" if f is not None else " (F assumed 1 as none was given)")
                ),
            })

    elif endpoint == "cmax":
        mw = ctx.get("molecular_weight")
        if mw:
            molar = (value * 1e-9) / mw * 1e3  # ng/mL -> g/L -> mol/L
            out.append({
                "label": "Molar peak concentration",
                "value": _concentration_phrase(molar),
                "how": f"{_fmt(value)} ng/mL ÷ {_fmt(mw)} g/mol",
            })
            fu = ctx.get("unbound_fraction")
            if fu:
                out.append({
                    "label": "Unbound peak concentration",
                    "value": _concentration_phrase(molar * fu),
                    "how": (
                        f"{_fmt(fu * 100)}% unbound; only this fraction is "
                        "available to bind the target"
                    ),
                })

    elif endpoint == "mae":
        out.append({
            "label": "Typical error as a fold-change",
            "value": f"{_fmt(10 ** value)}-fold",
            "how": f"10^{_fmt(value)} in the underlying linear quantity",
        })
        noise = ctx.get("experimental_noise_floor")
        if noise:
            out.append({
                "label": "Against the experimental noise floor",
                "value": (
                    f"{_fmt(value / noise)}x the measurement disagreement"
                    if noise > 0 else "no floor estimate"
                ),
                "how": (
                    f"repeat measurements of the same molecule differ by about "
                    f"{_fmt(noise)} log units, which bounds achievable accuracy"
                ),
            })

    elif endpoint == "coverage":
        nominal = ctx.get("nominal_coverage")
        if nominal:
            gap = value - nominal
            out.append({
                "label": "Against nominal",
                "value": f"{gap:+.3f}",
                "how": (
                    f"measured {_fmt(value)} against a nominal {_fmt(nominal)}; "
                    + ("intervals are too narrow" if gap < -0.03
                       else "intervals are wider than needed" if gap > 0.03
                       else "within 3 points of nominal")
                ),
            })

    elif endpoint == "ki":
        if value > 0:
            out.append({
                "label": "Equivalent pKi",
                "value": _fmt(9 - math.log10(value)),
                "how": f"pKi = 9 − log10({_fmt(value)} nM)",
            })

    return out


def concentration_vs_affinity(
    cmax_ng_per_ml: float, pki: float, molecular_weight: float,
    *, unbound_fraction: float | None = None,
) -> dict[str, Any]:
    """Relate a peak concentration to a measured affinity.

    This is the bridge from exposure to target engagement, and it is the one
    cross-endpoint inference worth computing automatically. It is reported with
    the caveats that make it honest: plasma is not the target tissue, total is
    not free, and a ratio above one does not establish a functional effect.
    """
    molar = (cmax_ng_per_ml * 1e-9) / molecular_weight * 1e3
    ki_molar = 10 ** (-pki)
    free = molar * unbound_fraction if unbound_fraction else molar
    ratio = free / ki_molar if ki_molar > 0 else float("inf")
    occupancy = free / (free + ki_molar) if (free + ki_molar) > 0 else 0.0
    return {
        "peak_concentration": _concentration_phrase(molar),
        "unbound_peak_concentration": (
            _concentration_phrase(free) if unbound_fraction else None),
        "ki": _concentration_phrase(ki_molar),
        "ratio_to_ki": ratio,
        "ratio_phrase": (
            f"peak {'unbound ' if unbound_fraction else ''}concentration is "
            f"{_fmt(ratio)}x the measured Ki"
        ),
        "implied_occupancy_at_peak": occupancy,
        "occupancy_phrase": (
            f"an equilibrium occupancy model gives {occupancy * 100:.0f}% of "
            "receptors bound at the peak"
        ),
        "caveats": [
            "Plasma concentration is not target-tissue concentration; the two "
            "can differ by orders of magnitude.",
            "Occupancy is computed from an equilibrium model and assumes the "
            "free concentration at the receptor equals the free plasma "
            "concentration.",
            "Occupancy is not a functional response. Receptor reserve can make "
            "low occupancy sufficient, and an antagonist occupies without "
            "activating.",
            "The Ki was measured in a binding assay whose conditions differ "
            "from the in vivo environment.",
        ],
    }


def interpret(
    endpoint: str, value: float | None, context: dict[str, Any] | None = None
) -> Interpretation | None:
    """Explain one reported quantity, using its value and the model's metadata."""
    definition: EndpointDefinition | None = get_definition(endpoint)
    if definition is None:
        return None
    ctx = dict(context or {})

    interp = Interpretation(
        endpoint=definition.key, symbol=definition.symbol, name=definition.name,
        value=value, unit=ctx.get("unit") or definition.unit,
    )

    interp.what_is_measured = {
        "definition": definition.definition,
        "unit": interp.unit,
        "target": ctx.get("target"),
        "species": ctx.get("species"),
        "experimental_system": ctx.get("system"),
        "source_kind": ctx.get("source_kind"),
        "biological_context": ctx.get("biological_context"),
    }

    interp.mathematics = {
        "transformation": definition.transformation,
        "stated_value": None if value is None else f"{_fmt(value)} {interp.unit}",
    }
    interp.derived = _derive(endpoint, value, ctx) if value is not None else []

    interp.biological_meaning = {
        "establishes": definition.establishes,
        "does_not_establish": definition.does_not_establish,
        "common_misreadings": definition.common_misreadings,
    }

    drivers = list(definition.uncertainty_drivers)
    dominant = ctx.get("dominant_uncertainty")
    if dominant:
        drivers.insert(0, dominant)
    interp.uncertainty = {
        "interval_low": ctx.get("interval_low"),
        "interval_high": ctx.get("interval_high"),
        "interval_nominal": ctx.get("interval_nominal"),
        "interval_measured_coverage": ctx.get("interval_measured_coverage"),
        "model_performance": ctx.get("model_performance"),
        "applicability": ctx.get("applicability"),
        "missing_parameters": ctx.get("missing_parameters", []),
        "drivers": drivers,
        "dominant_driver": dominant,
    }

    interp.why_this_result = list(ctx.get("contributions", []))

    interp.validation = {
        "measurements_that_would_test_it": definition.validating_measurements,
        "open_assumptions": ctx.get("assumptions", []),
    }
    return interp
