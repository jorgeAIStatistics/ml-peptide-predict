"""Dimensional analysis and unit normalization for quantitative evidence.

Every number that enters ATLAS from the literature carries a unit string written
by a human ("ng/mL", "L/h/kg", "nmol/L"). Pharmacokinetic arithmetic silently
produces garbage when those are mixed up, so quantities are parsed into an
explicit SI dimension vector and all arithmetic is dimension-checked.

Molar <-> mass conversion is deliberately not automatic: it requires a molecular
weight, which for several entries in the registry is unknown. Callers must pass
``mw_g_per_mol`` explicitly, and a missing MW raises rather than defaulting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = [
    "Dimension",
    "Quantity",
    "UnitError",
    "DimensionError",
    "parse_unit",
    "quantity",
    "convert",
    "DIMENSIONLESS",
    "MASS",
    "VOLUME",
    "TIME",
    "SUBSTANCE",
    "MASS_CONCENTRATION",
    "MOLAR_CONCENTRATION",
    "CLEARANCE",
]


class UnitError(ValueError):
    """Raised when a unit string cannot be parsed."""


class DimensionError(ValueError):
    """Raised when an operation combines incompatible dimensions."""


@dataclass(frozen=True, slots=True)
class Dimension:
    """Exponents over SI base dimensions used in pharmacology.

    Only the four bases that appear in PK/PD work are tracked. Exponents are
    integers; no fractional dimensions arise in this domain.
    """

    mass: int = 0
    length: int = 0
    time: int = 0
    substance: int = 0

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            self.mass + other.mass,
            self.length + other.length,
            self.time + other.time,
            self.substance + other.substance,
        )

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            self.mass - other.mass,
            self.length - other.length,
            self.time - other.time,
            self.substance - other.substance,
        )

    def __pow__(self, n: int) -> "Dimension":
        return Dimension(
            self.mass * n, self.length * n, self.time * n, self.substance * n
        )

    @property
    def is_dimensionless(self) -> bool:
        return self == DIMENSIONLESS

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "1"
        parts = []
        for name, exp in (
            ("kg", self.mass),
            ("m", self.length),
            ("s", self.time),
            ("mol", self.substance),
        ):
            if exp == 1:
                parts.append(name)
            elif exp != 0:
                parts.append(f"{name}^{exp}")
        return "·".join(parts)


DIMENSIONLESS: Final = Dimension()
MASS: Final = Dimension(mass=1)
LENGTH: Final = Dimension(length=1)
TIME: Final = Dimension(time=1)
SUBSTANCE: Final = Dimension(substance=1)
VOLUME: Final = Dimension(length=3)
MASS_CONCENTRATION: Final = MASS / VOLUME
MOLAR_CONCENTRATION: Final = SUBSTANCE / VOLUME
CLEARANCE: Final = VOLUME / TIME

# Atomic unit table: symbol -> (multiplier to SI, dimension).
# Keys are matched case-sensitively first, then case-insensitively, because
# "mg" and "Mg" mean different things in principle but literature uses "ML"
# and "mL" interchangeably for millilitre.
_ATOMIC: Final[dict[str, tuple[float, Dimension]]] = {
    # mass
    "kg": (1.0, MASS),
    "g": (1e-3, MASS),
    "mg": (1e-6, MASS),
    "ug": (1e-9, MASS),
    "µg": (1e-9, MASS),  # micro sign
    "μg": (1e-9, MASS),  # greek mu
    "mcg": (1e-9, MASS),
    "ng": (1e-12, MASS),
    "pg": (1e-15, MASS),
    "Da": (1.66053906892e-27, MASS),
    "kDa": (1.66053906892e-24, MASS),
    # substance
    "mol": (1.0, SUBSTANCE),
    "mmol": (1e-3, SUBSTANCE),
    "umol": (1e-6, SUBSTANCE),
    "µmol": (1e-6, SUBSTANCE),
    "μmol": (1e-6, SUBSTANCE),
    "nmol": (1e-9, SUBSTANCE),
    "pmol": (1e-12, SUBSTANCE),
    "fmol": (1e-15, SUBSTANCE),
    # volume (SI base is m^3)
    "m3": (1.0, VOLUME),
    "L": (1e-3, VOLUME),
    "dL": (1e-4, VOLUME),
    "mL": (1e-6, VOLUME),
    "uL": (1e-9, VOLUME),
    "µL": (1e-9, VOLUME),
    "μL": (1e-9, VOLUME),
    "nL": (1e-12, VOLUME),
    # time
    "s": (1.0, TIME),
    "sec": (1.0, TIME),
    "min": (60.0, TIME),
    "h": (3600.0, TIME),
    "hr": (3600.0, TIME),
    "d": (86400.0, TIME),
    "day": (86400.0, TIME),
    "wk": (604800.0, TIME),
    "week": (604800.0, TIME),
    # length
    "m": (1.0, LENGTH),
    "cm": (1e-2, LENGTH),
    "mm": (1e-3, LENGTH),
    # dimensionless
    "1": (1.0, DIMENSIONLESS),
    "": (1.0, DIMENSIONLESS),
    "%": (1e-2, DIMENSIONLESS),
    "fraction": (1.0, DIMENSIONLESS),
    "ratio": (1.0, DIMENSIONLESS),
}

# Case-insensitive fallback, built once. Ambiguous collisions are dropped so a
# lookup that would be ambiguous fails loudly instead of guessing.
_CI_INDEX: Final[dict[str, str]] = {}
for _sym in _ATOMIC:
    _key = _sym.casefold()
    _CI_INDEX[_key] = "<ambiguous>" if _key in _CI_INDEX else _sym
_CI_INDEX = {k: v for k, v in _CI_INDEX.items() if v != "<ambiguous>"}

_TOKEN_RE: Final = re.compile(
    r"^(?P<sym>[A-Za-zµμ%1][A-Za-z0-9µμ%]*?)"
    r"(?:\^?(?P<exp>-?[0-9]+))?$"
)


def _lookup_atomic(symbol: str) -> tuple[float, Dimension]:
    if symbol in _ATOMIC:
        return _ATOMIC[symbol]
    folded = symbol.casefold()
    if folded in _CI_INDEX:
        return _ATOMIC[_CI_INDEX[folded]]
    raise UnitError(f"unknown unit symbol: {symbol!r}")


def _parse_token(token: str) -> tuple[float, Dimension]:
    """Parse one factor such as ``mL``, ``L^2`` or ``h-1``."""
    token = token.strip()
    if not token:
        return 1.0, DIMENSIONLESS
    # A bare exponent suffix on a multi-char symbol ("m3") is ambiguous with
    # symbols that legitimately contain digits; none do here, so trailing
    # digits are always exponents except for the explicit "m3" entry.
    if token in _ATOMIC:
        return _ATOMIC[token]
    match = _TOKEN_RE.match(token)
    if not match:
        raise UnitError(f"cannot parse unit factor: {token!r}")
    factor, dim = _lookup_atomic(match.group("sym"))
    exp = int(match.group("exp") or 1)
    return factor**exp, dim**exp


def parse_unit(unit: str) -> tuple[float, Dimension]:
    """Return ``(multiplier_to_SI, dimension)`` for a composite unit string.

    Accepts ``/`` for division, ``*`` or ``.`` or ``·`` for multiplication, and
    repeated division as in ``mL/min/kg`` (read left-to-right, so every factor
    after the first slash divides).
    """
    if unit is None:
        raise UnitError("unit string is None")
    text = unit.strip()
    if not text:
        return 1.0, DIMENSIONLESS
    text = text.replace("·", "*").replace("⋅", "*").replace(".", "*")
    # Split on slashes first: everything after the first slash is a denominator.
    segments = text.split("/")
    numerator, denominators = segments[0], segments[1:]

    total_factor = 1.0
    total_dim = DIMENSIONLESS
    for token in numerator.split("*"):
        f, d = _parse_token(token)
        total_factor *= f
        total_dim = total_dim * d
    for segment in denominators:
        for token in segment.split("*"):
            f, d = _parse_token(token)
            total_factor /= f
            total_dim = total_dim / d
    return total_factor, total_dim


@dataclass(frozen=True, slots=True)
class Quantity:
    """A value with a unit, stored alongside its SI magnitude and dimension.

    The original unit string is preserved so that evidence can be displayed in
    the units the source actually reported.
    """

    value: float
    unit: str
    si_value: float
    dimension: Dimension

    def __str__(self) -> str:
        return f"{self.value:g} {self.unit}" if self.unit else f"{self.value:g}"

    # -- arithmetic ---------------------------------------------------------
    def _require_same_dimension(self, other: "Quantity", op: str) -> None:
        if self.dimension != other.dimension:
            raise DimensionError(
                f"cannot {op} {self.unit or '1'} [{self.dimension}] and "
                f"{other.unit or '1'} [{other.dimension}]"
            )

    def __add__(self, other: "Quantity") -> "Quantity":
        self._require_same_dimension(other, "add")
        return _from_si(self.si_value + other.si_value, self.unit, self.dimension)

    def __sub__(self, other: "Quantity") -> "Quantity":
        self._require_same_dimension(other, "subtract")
        return _from_si(self.si_value - other.si_value, self.unit, self.dimension)

    def __mul__(self, other: "Quantity | float | int") -> "Quantity":
        if isinstance(other, (int, float)):
            return Quantity(
                self.value * other, self.unit, self.si_value * other, self.dimension
            )
        unit = _compose_unit(self.unit, other.unit, "*")
        dim = self.dimension * other.dimension
        return _from_si(self.si_value * other.si_value, unit, dim)

    __rmul__ = __mul__

    def __truediv__(self, other: "Quantity | float | int") -> "Quantity":
        if isinstance(other, (int, float)):
            return Quantity(
                self.value / other, self.unit, self.si_value / other, self.dimension
            )
        unit = _compose_unit(self.unit, other.unit, "/")
        dim = self.dimension / other.dimension
        return _from_si(self.si_value / other.si_value, unit, dim)

    def to(self, unit: str, *, mw_g_per_mol: float | None = None) -> "Quantity":
        return convert(self, unit, mw_g_per_mol=mw_g_per_mol)


def _compose_unit(left: str, right: str, op: str) -> str:
    left = left or "1"
    right = right or "1"
    if op == "*":
        if left == "1":
            return right
        if right == "1":
            return left
        return f"{left}*{right}"
    if left == "1":
        return f"1/{right}" if right != "1" else "1"
    if right == "1":
        return left
    return f"{left}/({right})"


def _from_si(si_value: float, unit: str, dimension: Dimension) -> Quantity:
    """Rebuild a Quantity in ``unit`` from an SI magnitude."""
    try:
        factor, parsed_dim = parse_unit(unit)
    except UnitError:
        # Composite units synthesised by arithmetic may not re-parse (e.g.
        # "ng*h/(mL)"); fall back to reporting the SI magnitude directly.
        return Quantity(si_value, unit, si_value, dimension)
    if parsed_dim != dimension:
        return Quantity(si_value, unit, si_value, dimension)
    return Quantity(si_value / factor, unit, si_value, dimension)


def quantity(value: float, unit: str) -> Quantity:
    """Construct a :class:`Quantity` from a value and a unit string."""
    factor, dim = parse_unit(unit)
    return Quantity(float(value), unit.strip(), float(value) * factor, dim)


def convert(
    q: Quantity, target_unit: str, *, mw_g_per_mol: float | None = None
) -> Quantity:
    """Convert ``q`` to ``target_unit``.

    Mass <-> substance conversion (e.g. ``ng/mL`` to ``nmol/L``) requires a
    molecular weight in g/mol. Omitting it raises :class:`DimensionError`
    rather than silently returning an unconverted value.
    """
    target_factor, target_dim = parse_unit(target_unit)
    si = q.si_value
    source_dim = q.dimension

    if source_dim != target_dim:
        # The only legitimate bridge in this domain is mass <-> amount of
        # substance via molecular weight, at matching exponent.
        delta = target_dim / source_dim
        bridge = (SUBSTANCE / MASS) ** 1
        if delta == bridge or delta == bridge**-1:
            if mw_g_per_mol is None:
                raise DimensionError(
                    f"converting {q.unit!r} [{source_dim}] to {target_unit!r} "
                    f"[{target_dim}] requires a molecular weight; none supplied"
                )
            if mw_g_per_mol <= 0:
                raise ValueError("mw_g_per_mol must be positive")
            mw_si = mw_g_per_mol * 1e-3  # kg/mol
            si = si / mw_si if delta == bridge else si * mw_si
        else:
            raise DimensionError(
                f"incompatible units: {q.unit!r} [{source_dim}] -> "
                f"{target_unit!r} [{target_dim}]"
            )

    return Quantity(si / target_factor, target_unit.strip(), si, target_dim)
