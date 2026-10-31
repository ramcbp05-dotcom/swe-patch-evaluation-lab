import pytest

from fieldlab import units


class TestToSi:
    @pytest.mark.parametrize("value,unit,want", [
        (0.0, "C", 273.15),
        (100.0, "C", 373.15),
        (32.0, "F", 273.15),
        (212.0, "F", 373.15),
        (-40.0, "F", 233.15),
        (0.0, "K", 0.0),
        (300.0, "K", 300.0),
    ])
    def test_temperature_is_affine(self, value, unit, want):
        assert units.to_si("temperature", value, unit) == pytest.approx(want, abs=1e-9)

    def test_fahrenheit_and_celsius_meet_at_minus_40(self):
        assert units.to_si("temperature", -40.0, "F") == pytest.approx(
            units.to_si("temperature", -40.0, "C"), abs=1e-9
        )

    def test_temperature_difference_scales_by_five_ninths(self):
        """An affine map has a slope: 180 degF of span is 100 K of span."""
        hot = units.to_si("temperature", 212.0, "F")
        cold = units.to_si("temperature", 32.0, "F")
        assert hot - cold == pytest.approx(100.0, rel=1e-12)

    @pytest.mark.parametrize("value,unit,want", [
        (1.0, "Pa", 1.0),
        (1.0, "kPa", 1000.0),
        (1.0, "MPa", 1.0e6),
        (1.0, "bar", 100000.0),
        (1.0, "atm", 101325.0),
        (1.0, "psi", 6894.757293168361),
    ])
    def test_pressure(self, value, unit, want):
        assert units.to_si("pressure", value, unit) == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("value,unit,want", [
        (1.0, "m", 1.0),
        (1.0, "km", 1000.0),
        (1.0, "cm", 0.01),
        (1.0, "mm", 0.001),
        (1.0, "um", 1.0e-6),
        (1.0, "in", 0.0254),
        (1.0, "ft", 0.3048),
    ])
    def test_length(self, value, unit, want):
        assert units.to_si("length", value, unit) == pytest.approx(want, rel=1e-12)

    def test_twelve_inches_make_a_foot(self):
        assert units.to_si("length", 12.0, "in") == pytest.approx(
            units.to_si("length", 1.0, "ft"), rel=1e-12
        )

    def test_unknown_quantity_raises(self):
        with pytest.raises(ValueError, match="unknown quantity"):
            units.to_si("charge", 1.0, "C")

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            units.to_si("length", 1.0, "furlong")

    def test_unit_from_another_quantity_is_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            units.to_si("length", 1.0, "psi")


class TestSupportedUnits:
    def test_lists_units(self):
        assert "psi" in units.supported_units("pressure")
        assert "F" in units.supported_units("temperature")

    def test_unknown_quantity_raises(self):
        with pytest.raises(ValueError, match="unknown quantity"):
            units.supported_units("luminance")
