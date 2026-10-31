"""Fail-to-pass tests for affine-unit-conversion.

An affine map has two free parameters, so it takes two anchors to pin down.
Every wrong candidate in this case satisfies one of them.
"""

import pytest

from fieldlab.units import supported_units, to_si


class TestCelsius:
    @pytest.mark.parametrize("value,want", [
        (0.0, 273.15), (1.0, 274.15), (100.0, 373.15),
        (-273.15, 0.0), (25.0, 298.15), (-40.0, 233.15),
    ])
    def test_anchors(self, value, want):
        got = to_si("temperature", value, "C")
        assert got == pytest.approx(want, abs=1e-9), (
            f"{value} C -> {got}, expected {want}"
        )

    def test_slope_is_one(self):
        """A 1 degC step is a 1 K step."""
        step = to_si("temperature", 11.0, "C") - to_si("temperature", 10.0, "C")
        assert step == pytest.approx(1.0, rel=1e-12), (
            f"a 1 degC step came out as {step} K"
        )

    def test_absolute_zero(self):
        assert to_si("temperature", -273.15, "C") == pytest.approx(0.0, abs=1e-9)


class TestFahrenheit:
    @pytest.mark.parametrize("value,want", [
        (32.0, 273.15), (212.0, 373.15), (-40.0, 233.15),
        (0.0, 255.3722222222222), (98.6, 310.15), (-459.67, 0.0),
    ])
    def test_anchors(self, value, want):
        got = to_si("temperature", value, "F")
        assert got == pytest.approx(want, abs=1e-6), (
            f"{value} F -> {got}, expected {want}"
        )

    def test_slope_is_five_ninths(self):
        """180 degF of span is 100 K of span. This is the missing slope."""
        span = to_si("temperature", 212.0, "F") - to_si("temperature", 32.0, "F")
        assert span == pytest.approx(100.0, rel=1e-12), (
            f"the 32..212 F span came out as {span} K, expected 100.0. "
            "A slope of 1 gives 180.0."
        )

    def test_one_degree_step(self):
        step = to_si("temperature", 1.0, "F") - to_si("temperature", 0.0, "F")
        assert step == pytest.approx(5.0 / 9.0, rel=1e-12), (
            f"a 1 degF step came out as {step} K, expected {5 / 9:.9f}"
        )

    def test_absolute_zero(self):
        assert to_si("temperature", -459.67, "F") == pytest.approx(0.0, abs=1e-6)


class TestScalesAgree:
    def test_celsius_and_fahrenheit_meet_at_minus_40(self):
        """The one temperature where the two scales read the same number."""
        f = to_si("temperature", -40.0, "F")
        c = to_si("temperature", -40.0, "C")
        assert f == pytest.approx(c, abs=1e-9), (
            f"-40 F gave {f} K but -40 C gave {c} K; the scales cross at -40"
        )

    @pytest.mark.parametrize("celsius", [-200.0, -40.0, 0.0, 37.0, 100.0, 1000.0])
    def test_round_trip_through_fahrenheit(self, celsius):
        fahrenheit = celsius * 9.0 / 5.0 + 32.0
        assert to_si("temperature", fahrenheit, "F") == pytest.approx(
            to_si("temperature", celsius, "C"), abs=1e-6
        ), f"{celsius} C and {fahrenheit} F must give the same kelvin value"

    def test_kelvin_is_the_identity(self):
        for value in (0.0, 1.0, 300.0, 5000.0):
            assert to_si("temperature", value, "K") == pytest.approx(value, rel=1e-14)

    def test_affine_not_linear(self):
        """A linear map sends 0 to 0; this one must not."""
        assert to_si("temperature", 0.0, "C") != pytest.approx(0.0, abs=1e-6), (
            "0 C mapped to 0 K, so the conversion is being applied as a pure "
            "scale factor"
        )


class TestOtherQuantitiesUntouched:
    @pytest.mark.parametrize("value,unit,want", [
        (1.0, "psi", 6894.757293168361), (1.0, "atm", 101325.0),
        (1.0, "kPa", 1000.0), (1.0, "bar", 100000.0),
    ])
    def test_pressure(self, value, unit, want):
        assert to_si("pressure", value, unit) == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("value,unit,want", [
        (1.0, "in", 0.0254), (1.0, "ft", 0.3048), (1.0, "cm", 0.01),
    ])
    def test_length(self, value, unit, want):
        assert to_si("length", value, unit) == pytest.approx(want, rel=1e-12)

    def test_pressure_and_length_are_purely_multiplicative(self):
        """Only temperature is affine: zero must map to zero for the others."""
        assert to_si("pressure", 0.0, "psi") == 0.0
        assert to_si("length", 0.0, "ft") == 0.0

    def test_unit_tables_are_unchanged(self):
        assert set(supported_units("temperature")) == {"K", "C", "F"}
        assert "psi" in supported_units("pressure"), (
            f"pressure units are {supported_units('pressure')}"
        )

    def test_unknown_unit_still_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            to_si("temperature", 1.0, "R")
