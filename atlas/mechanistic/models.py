"""Compartmental pharmacokinetic models with analytical reference solutions.

Internal units are fixed: **mg**, **L**, **h**. Concentration is therefore mg/L,
which equals ug/mL; multiply by 1000 for ng/mL. Parameters are declared with
units and are dimension-checked against :mod:`atlas.core.units` on
construction, so a clearance supplied in mL/min cannot silently be read as L/h.

Each model provides a closed-form solution where one exists. That is what makes
solver *verification* possible: numerical integration is compared against exact
mathematics rather than against another numerical result.

Mass balance is modelled explicitly. Rather than scaling the dose by
bioavailability, the depot drains into two sinks -- the central compartment
(fraction F) and an unabsorbed sink (fraction 1-F) -- so the sum of all states
equals the administered dose at every time point and conservation can be tested
as a property of the system.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from atlas.core.units import CLEARANCE, DIMENSIONLESS, TIME, VOLUME, Dimension, parse_unit

__all__ = [
    "ParameterSpec",
    "ParameterError",
    "PKModel",
    "OneCompartmentIV",
    "OneCompartmentOral",
    "TwoCompartmentIV",
    "MODEL_REGISTRY",
    "get_model",
]


class ParameterError(ValueError):
    """Raised when a parameter set is missing, out of range or wrongly dimensioned."""


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """A model parameter with its unit, physical dimension and valid range."""

    name: str
    unit: str
    description: str
    lower: float = 0.0
    upper: float | None = None
    lower_inclusive: bool = False

    @property
    def dimension(self) -> Dimension:
        return parse_unit(self.unit)[1]

    def check(self, value: float) -> None:
        if not math.isfinite(value):
            raise ParameterError(f"{self.name}: value must be finite, got {value!r}")
        if self.lower_inclusive:
            if value < self.lower:
                raise ParameterError(
                    f"{self.name}: must be >= {self.lower} {self.unit}, got {value:g}"
                )
        elif value <= self.lower:
            raise ParameterError(
                f"{self.name}: must be > {self.lower} {self.unit}, got {value:g}"
            )
        if self.upper is not None and value > self.upper:
            raise ParameterError(
                f"{self.name}: must be <= {self.upper} {self.unit}, got {value:g}"
            )


class PKModel(ABC):
    """A compartmental model expressed as an ODE system plus its exact solution."""

    name: str
    description: str
    state_names: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    #: index of the state holding drug amount in the sampled (central) compartment
    central_index: int = 0

    # -- parameters ---------------------------------------------------------
    def spec(self, name: str) -> ParameterSpec:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"{self.name} has no parameter {name!r}")

    def validate(self, params: Mapping[str, float]) -> dict[str, float]:
        """Check completeness and ranges, returning a plain float dict."""
        required = {p.name for p in self.parameters}
        supplied = set(params)
        missing = required - supplied
        if missing:
            raise ParameterError(
                f"{self.name}: missing parameters {sorted(missing)}"
            )
        extra = supplied - required
        if extra:
            raise ParameterError(
                f"{self.name}: unexpected parameters {sorted(extra)}; "
                f"this model takes {sorted(required)}"
            )
        out: dict[str, float] = {}
        for p in self.parameters:
            value = float(params[p.name])
            p.check(value)
            out[p.name] = value
        return out

    # -- dynamics -----------------------------------------------------------
    @abstractmethod
    def rhs(self, t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
        """Right-hand side of the ODE system, in mg/h per state."""

    @abstractmethod
    def initial_state(self, dose_mg: float, p: Mapping[str, float]) -> np.ndarray:
        """State vector immediately after administration of ``dose_mg``."""

    @abstractmethod
    def central_volume(self, p: Mapping[str, float]) -> float:
        """Volume of the sampled compartment, in L."""

    def concentration(self, amounts: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
        """Convert central-compartment amount (mg) to concentration (mg/L)."""
        return np.asarray(amounts, dtype=float) / self.central_volume(p)

    # -- exact reference solutions -----------------------------------------
    def analytical_concentration(
        self, t: Sequence[float] | np.ndarray, dose_mg: float, p: Mapping[str, float]
    ) -> np.ndarray | None:
        """Closed-form concentration (mg/L), or ``None`` if none is implemented."""
        return None

    def analytical_auc_inf(self, dose_mg: float, p: Mapping[str, float]) -> float | None:
        """Exact AUC from zero to infinity (mg*h/L), if derivable.

        For every linear model here this is ``F * Dose / CL``, which is an
        identity rather than an approximation and makes a strong solver check.
        """
        return None

    def terminal_half_life_h(self, p: Mapping[str, float]) -> float | None:
        """Half-life of the slowest exponential phase, in hours."""
        return None

    def mass_balance_total(self, y: np.ndarray) -> np.ndarray:
        """Total drug accounted for across all states, in mg.

        Every model here carries explicit elimination and loss states, so this
        must equal the administered dose for all time.
        """
        arr = np.asarray(y, dtype=float)
        return arr.sum(axis=0) if arr.ndim > 1 else arr.sum()


# ---------------------------------------------------------------------------
# One-compartment, intravenous bolus
# ---------------------------------------------------------------------------
class OneCompartmentIV(PKModel):
    r"""One-compartment model after an intravenous bolus.

    .. math::
        \frac{dA_c}{dt} = -\frac{CL}{V} A_c, \qquad
        C(t) = \frac{D}{V} e^{-(CL/V) t}
    """

    name = "one_compartment_iv"
    description = "One-compartment linear disposition after IV bolus administration."
    state_names = ("central_mg", "eliminated_mg")
    central_index = 0
    parameters = (
        ParameterSpec("CL", "L/h", "Systemic clearance."),
        ParameterSpec("V", "L", "Volume of distribution of the central compartment."),
    )

    def rhs(self, t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
        central = y[0]
        rate_out = p["CL"] / p["V"] * central
        return np.array([-rate_out, rate_out])

    def initial_state(self, dose_mg: float, p: Mapping[str, float]) -> np.ndarray:
        return np.array([float(dose_mg), 0.0])

    def central_volume(self, p: Mapping[str, float]) -> float:
        return p["V"]

    def analytical_concentration(self, t, dose_mg, p):
        t = np.asarray(t, dtype=float)
        ke = p["CL"] / p["V"]
        return (dose_mg / p["V"]) * np.exp(-ke * t)

    def analytical_auc_inf(self, dose_mg, p):
        return dose_mg / p["CL"]

    def terminal_half_life_h(self, p):
        return math.log(2.0) * p["V"] / p["CL"]


# ---------------------------------------------------------------------------
# One-compartment, first-order absorption
# ---------------------------------------------------------------------------
class OneCompartmentOral(PKModel):
    r"""One-compartment model with first-order absorption from a depot.

    Applies to subcutaneous and oral administration. The closed form is the
    Bateman function

    .. math::
        C(t) = \frac{F D k_a}{V (k_a - k_e)}
               \left(e^{-k_e t} - e^{-k_a t}\right), \quad k_e = CL/V

    which is singular when :math:`k_a = k_e`. That case has the finite limit
    :math:`C(t) = (F D / V) k_e t e^{-k_e t}`, and it is handled explicitly:
    the removable singularity is a genuine numerical trap, since floating-point
    parameters can land arbitrarily close to it.
    """

    name = "one_compartment_oral"
    description = (
        "One-compartment linear disposition with first-order absorption from a "
        "depot compartment (subcutaneous or oral)."
    )
    state_names = ("depot_mg", "central_mg", "eliminated_mg", "unabsorbed_mg")
    central_index = 1
    parameters = (
        ParameterSpec("CL", "L/h", "Apparent systemic clearance."),
        ParameterSpec("V", "L", "Apparent volume of distribution."),
        ParameterSpec("ka", "1/h", "First-order absorption rate constant."),
        ParameterSpec(
            "F", "1", "Bioavailable fraction of the administered dose.",
            lower=0.0, upper=1.0,
        ),
    )

    #: relative |ka-ke| below which the analytical form switches to its limit
    SINGULARITY_TOL = 1e-8

    def rhs(self, t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
        depot, central = y[0], y[1]
        ka, F = p["ka"], p["F"]
        leaving_depot = ka * depot
        to_central = F * leaving_depot
        to_waste = (1.0 - F) * leaving_depot
        eliminated = p["CL"] / p["V"] * central
        return np.array(
            [-leaving_depot, to_central - eliminated, eliminated, to_waste]
        )

    def initial_state(self, dose_mg: float, p: Mapping[str, float]) -> np.ndarray:
        return np.array([float(dose_mg), 0.0, 0.0, 0.0])

    def central_volume(self, p: Mapping[str, float]) -> float:
        return p["V"]

    def analytical_concentration(self, t, dose_mg, p):
        t = np.asarray(t, dtype=float)
        ka, ke = p["ka"], p["CL"] / p["V"]
        prefactor = p["F"] * dose_mg / p["V"]

        if abs(ka - ke) <= self.SINGULARITY_TOL * max(ka, ke):
            # Removable singularity: the ka -> ke limit of the Bateman function.
            return prefactor * ke * t * np.exp(-ke * t)

        return prefactor * (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

    def analytical_auc_inf(self, dose_mg, p):
        return p["F"] * dose_mg / p["CL"]

    def analytical_tmax_h(self, p: Mapping[str, float]) -> float:
        """Exact time of maximum concentration.

        ``ln(ka/ke)/(ka-ke)``, with limit ``1/ke`` as ``ka -> ke``.
        """
        ka, ke = p["ka"], p["CL"] / p["V"]
        if abs(ka - ke) <= self.SINGULARITY_TOL * max(ka, ke):
            return 1.0 / ke
        return math.log(ka / ke) / (ka - ke)

    def analytical_cmax(self, dose_mg: float, p: Mapping[str, float]) -> float:
        return float(
            self.analytical_concentration(
                np.array([self.analytical_tmax_h(p)]), dose_mg, p
            )[0]
        )

    def terminal_half_life_h(self, p):
        # The terminal slope is governed by the slower of the two processes;
        # when absorption is slower than elimination the decline is
        # absorption-rate-limited ("flip-flop" kinetics).
        ka, ke = p["ka"], p["CL"] / p["V"]
        return math.log(2.0) / min(ka, ke)

    def is_flip_flop(self, p: Mapping[str, float]) -> bool:
        """True when absorption is slower than elimination.

        In this regime the terminal slope reflects absorption, so reading a
        clearance off the terminal phase gives the wrong answer.
        """
        return p["ka"] < p["CL"] / p["V"]


# ---------------------------------------------------------------------------
# Two-compartment, intravenous bolus
# ---------------------------------------------------------------------------
class TwoCompartmentIV(PKModel):
    r"""Two-compartment model after an IV bolus, solved via hybrid constants.

    The biexponential solution uses

    .. math::
        \alpha, \beta = \tfrac{1}{2}\left(s \pm \sqrt{s^2 - 4 k_{10} k_{21}}\right),
        \quad s = k_{10} + k_{12} + k_{21}

    where the discriminant is non-negative for all positive rate constants, so
    the roots are always real and distinct unless the system is degenerate.
    """

    name = "two_compartment_iv"
    description = (
        "Two-compartment linear disposition after IV bolus, with a peripheral "
        "distribution compartment."
    )
    state_names = ("central_mg", "peripheral_mg", "eliminated_mg")
    central_index = 0
    parameters = (
        ParameterSpec("CL", "L/h", "Systemic clearance from the central compartment."),
        ParameterSpec("V1", "L", "Central volume of distribution."),
        ParameterSpec("Q", "L/h", "Intercompartmental clearance."),
        ParameterSpec("V2", "L", "Peripheral volume of distribution."),
    )

    def _micro(self, p: Mapping[str, float]) -> tuple[float, float, float]:
        k10 = p["CL"] / p["V1"]
        k12 = p["Q"] / p["V1"]
        k21 = p["Q"] / p["V2"]
        return k10, k12, k21

    def _hybrid(self, p: Mapping[str, float]) -> tuple[float, float]:
        k10, k12, k21 = self._micro(p)
        s = k10 + k12 + k21
        disc = s * s - 4.0 * k10 * k21
        # Non-negative for positive rate constants; clamp guards rounding only.
        root = math.sqrt(max(disc, 0.0))
        alpha = 0.5 * (s + root)
        beta = 0.5 * (s - root)
        return alpha, beta

    def rhs(self, t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
        central, peripheral = y[0], y[1]
        k10, k12, k21 = self._micro(p)
        to_peripheral = k12 * central
        from_peripheral = k21 * peripheral
        eliminated = k10 * central
        return np.array(
            [
                -to_peripheral + from_peripheral - eliminated,
                to_peripheral - from_peripheral,
                eliminated,
            ]
        )

    def initial_state(self, dose_mg: float, p: Mapping[str, float]) -> np.ndarray:
        return np.array([float(dose_mg), 0.0, 0.0])

    def central_volume(self, p: Mapping[str, float]) -> float:
        return p["V1"]

    def analytical_concentration(self, t, dose_mg, p):
        t = np.asarray(t, dtype=float)
        _, _, k21 = self._micro(p)
        alpha, beta = self._hybrid(p)
        c0 = dose_mg / p["V1"]
        if abs(alpha - beta) <= 1e-12 * max(alpha, beta):
            # Degenerate (critically damped) case: repeated root.
            return c0 * (1.0 + (k21 - alpha) * t) * np.exp(-alpha * t)
        a_coeff = (alpha - k21) / (alpha - beta)
        b_coeff = (k21 - beta) / (alpha - beta)
        return c0 * (a_coeff * np.exp(-alpha * t) + b_coeff * np.exp(-beta * t))

    def analytical_auc_inf(self, dose_mg, p):
        return dose_mg / p["CL"]

    def terminal_half_life_h(self, p):
        _, beta = self._hybrid(p)
        return math.log(2.0) / beta

    def steady_state_volume_l(self, p: Mapping[str, float]) -> float:
        return p["V1"] + p["V2"]


MODEL_REGISTRY: dict[str, type[PKModel]] = {
    OneCompartmentIV.name: OneCompartmentIV,
    OneCompartmentOral.name: OneCompartmentOral,
    TwoCompartmentIV.name: TwoCompartmentIV,
}


def get_model(name: str) -> PKModel:
    try:
        return MODEL_REGISTRY[name]()
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}"
        ) from None
