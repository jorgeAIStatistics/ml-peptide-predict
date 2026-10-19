"""Unit-system tests.

These check the properties that matter for PK correctness: that composite units
parse the way a pharmacologist writes them, that incompatible arithmetic raises
instead of producing a number, and that molar conversion refuses to guess a
molecular weight.
"""

from __future__ import annotations

import math

import pytest

from atlas.core.units import (
    CLEARANCE,
    DIMENSIONLESS,
    MASS_CONCENTRATION,
    MOLAR_CONCENTRATION,
    VOLUME,
    DimensionError,
    UnitError,
    convert,
    parse_unit,
    quantity,
)


def approx(a: float, b: float, rel: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel)


class TestParsing:
    def test_simple_mass(self):
        factor, dim = parse_unit("mg")
        assert approx(factor, 1e-6)
        assert dim.mass == 1 and dim.length == 0

    def test_mass_concentration(self):
        factor, dim = parse_unit("ng/mL")
        assert dim == MASS_CONCENTRATION
        # 1 ng/mL = 1e-12 kg / 1e-6 m^3 = 1e-6 kg/m^3
        assert approx(factor, 1e-6)

    def test_molar_concentration(self):
        _, dim = parse_unit("nmol/L")
        assert dim == MOLAR_CONCENTRATION

    def test_clearance(self):
        factor, dim = parse_unit("L/h")
        assert dim == CLEARANCE
        assert approx(factor, 1e-3 / 3600.0)

    def test_repeated_division_reads_left_to_right(self):
        """mL/min/kg means mL per minute per kilogram, not mL*kg/min."""
        _, dim = parse_unit("mL/min/kg")
        assert dim.length == 3
        assert dim.time == -1
        assert dim.mass == -1

    def test_auc_units(self):
        _, dim = parse_unit("ng*h/mL")
        assert dim.mass == 1 and dim.length == -3 and dim.time == 1

    def test_middot_multiplication(self):
        a = parse_unit("ng·h/mL")
        b = parse_unit("ng*h/mL")
        assert a == b

    def test_exponent_notation(self):
        _, dim = parse_unit("L^2")
        assert dim.length == 6

    def test_per_kg_volume_is_volume_over_mass(self):
        _, dim = parse_unit("L/kg")
        assert dim.length == 3 and dim.mass == -1

    def test_mass_per_mass_is_dimensionless(self):
        _, dim = parse_unit("mg/kg")
        assert dim == DIMENSIONLESS

    def test_percent(self):
        factor, dim = parse_unit("%")
        assert dim == DIMENSIONLESS
        assert approx(factor, 1e-2)

    def test_empty_is_dimensionless(self):
        assert parse_unit("") == (1.0, DIMENSIONLESS)

    def test_case_insensitive_fallback(self):
        """Literature writes millilitre as mL and ML interchangeably."""
        assert parse_unit("ML")[1] == VOLUME

    def test_micro_variants_agree(self):
        for sym in ("ug", "mcg", "µg", "μg"):
            factor, dim = parse_unit(sym)
            assert approx(factor, 1e-9), sym
            assert dim.mass == 1

    def test_unknown_unit_raises(self):
        with pytest.raises(UnitError):
            parse_unit("IU")

    def test_garbage_raises(self):
        with pytest.raises(UnitError):
            parse_unit("ng//mL^")


class TestArithmetic:
    def test_addition_same_dimension_different_scale(self):
        total = quantity(1.0, "g") + quantity(500.0, "mg")
        assert approx(total.si_value, 1.5e-3)

    def test_addition_incompatible_raises(self):
        with pytest.raises(DimensionError):
            quantity(1.0, "mg") + quantity(1.0, "mL")

    def test_subtraction_incompatible_raises(self):
        with pytest.raises(DimensionError):
            quantity(1.0, "h") - quantity(1.0, "L")

    def test_concentration_times_volume_is_mass(self):
        amount = quantity(10.0, "ng/mL") * quantity(5.0, "L")
        assert amount.dimension.mass == 1
        assert amount.dimension.length == 0
        # 10 ng/mL * 5 L = 10 ng/mL * 5000 mL = 50000 ng = 5e-8 kg
        assert approx(amount.si_value, 5e-8)

    def test_dose_over_volume_is_concentration(self):
        conc = quantity(1.0, "mg") / quantity(5.0, "L")
        assert conc.dimension == MASS_CONCENTRATION
        # 1 mg / 5 L = 0.2 mg/L = 200 ng/mL = 2e-4 kg/m^3
        assert approx(conc.si_value, 2e-4)

    def test_clearance_from_dose_and_auc(self):
        """CL = Dose / AUC is the canonical identity; dimensions must work out."""
        dose = quantity(1.0, "mg")
        auc = quantity(100.0, "ng*h/mL")
        cl = dose / auc
        assert cl.dimension == CLEARANCE

    def test_scalar_multiplication_preserves_unit(self):
        q = quantity(2.0, "mg") * 3
        assert q.unit == "mg"
        assert approx(q.value, 6.0)


class TestConversion:
    def test_mass_concentration_roundtrip(self):
        q = quantity(1.0, "ng/mL")
        assert approx(convert(q, "ug/L").value, 1.0)

    def test_half_life_hours_to_days(self):
        q = quantity(168.0, "h")
        assert approx(convert(q, "d").value, 7.0)

    def test_molar_conversion_requires_mw(self):
        q = quantity(1.0, "ng/mL")
        with pytest.raises(DimensionError, match="molecular weight"):
            convert(q, "nmol/L")

    def test_molar_conversion_with_mw(self):
        """Semaglutide MW ~4113.6 g/mol: 41.136 ng/mL == 10 pmol/mL == 10 nmol/L."""
        q = quantity(41.136, "ng/mL")
        out = convert(q, "nmol/L", mw_g_per_mol=4113.6)
        assert approx(out.value, 10.0, rel=1e-9)

    def test_molar_conversion_reverse(self):
        q = quantity(10.0, "nmol/L")
        out = convert(q, "ng/mL", mw_g_per_mol=4113.6)
        assert approx(out.value, 41.136, rel=1e-9)

    def test_amount_mass_to_moles(self):
        q = quantity(1.0, "mg")
        out = convert(q, "nmol", mw_g_per_mol=1000.0)
        # 1 mg / 1000 g/mol = 1e-6 mol = 1000 nmol
        assert approx(out.value, 1000.0, rel=1e-9)

    def test_nonpositive_mw_rejected(self):
        with pytest.raises(ValueError):
            convert(quantity(1.0, "ng/mL"), "nmol/L", mw_g_per_mol=0.0)

    def test_truly_incompatible_conversion_raises(self):
        with pytest.raises(DimensionError, match="incompatible"):
            convert(quantity(1.0, "h"), "mL", mw_g_per_mol=100.0)

    def test_dalton_to_gram_per_mole_scale(self):
        """A 4113.6 Da molecule weighs 4113.6 g/mol; check Da is a real mass."""
        q = quantity(4113.6, "Da")
        assert q.dimension.mass == 1
        grams_per_mole = q.si_value * 6.02214076e23 * 1e3
        assert approx(grams_per_mole, 4113.6, rel=1e-6)
