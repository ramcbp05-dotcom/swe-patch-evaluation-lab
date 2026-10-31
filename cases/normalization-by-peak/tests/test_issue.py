"""Fail-to-pass tests for normalization-by-peak."""

import numpy as np
import pytest

from fieldlab.normalize import unit_area


class TestUnitAreaInvariant:
    def test_reported_case_narrow_and_wide(self):
        x = np.linspace(-20.0, 20.0, 2001)
        for width in (0.5, 5.0):
            y = np.exp(-0.5 * (x / width) ** 2)
            got = float(np.trapezoid(unit_area(x, y), x))
            assert got == pytest.approx(1.0, rel=1e-9), (
                f"width {width}: integral is {got:.6f}, expected 1.0"
            )

    def test_areas_are_width_independent(self):
        """The whole point: two different widths must give the same area."""
        x = np.linspace(-30.0, 30.0, 4001)
        areas = []
        for width in (0.4, 1.0, 3.0, 9.0):
            y = np.exp(-0.5 * (x / width) ** 2)
            areas.append(float(np.trapezoid(unit_area(x, y), x)))
        assert max(areas) - min(areas) < 1e-9, (
            f"integrals across widths: {areas}. They must all be 1.0"
        )

    def test_non_uniform_grid(self):
        rng = np.random.default_rng(11)
        x = np.sort(rng.uniform(0.0, 10.0, size=500))
        y = 1.0 + 4.0 * np.exp(-0.5 * ((x - 6.0) / 1.3) ** 2)
        got = float(np.trapezoid(unit_area(x, y), x))
        assert got == pytest.approx(1.0, rel=1e-10), (
            f"integral on a non-uniform grid is {got:.9f}"
        )

    @pytest.mark.parametrize("dx", [0.001, 0.1, 1.0, 7.0, 250.0])
    def test_grid_spacing_does_not_matter(self, dx):
        """A fix that forgets the x argument is correct only at dx == 1."""
        x = dx * np.arange(400, dtype=float)
        y = np.exp(-0.5 * ((x - x.mean()) / (40.0 * dx)) ** 2)
        got = float(np.trapezoid(unit_area(x, y), x))
        assert got == pytest.approx(1.0, rel=1e-9), (
            f"dx={dx}: integral is {got:.9g}. A factor of {dx} suggests the x grid "
            "is being ignored by the integration."
        )

    def test_endpoint_weighting_is_trapezoidal(self):
        """A rectangle-rule fix mis-weights the two end samples."""
        x = np.linspace(0.0, 1.0, 11)
        y = np.ones_like(x)
        out = unit_area(x, y)
        assert float(np.trapezoid(out, x)) == pytest.approx(1.0, rel=1e-12)
        # trapezoid of a constant 1 over [0, 1] is 1, so the scale factor is 1.
        np.testing.assert_allclose(out, 1.0, rtol=1e-12)

    def test_scale_invariance(self):
        x = np.linspace(0.0, 5.0, 301)
        y = np.exp(-((x - 2.0) ** 2))
        np.testing.assert_allclose(
            unit_area(x, y), unit_area(x, 1000.0 * y), rtol=1e-12
        )

    def test_input_is_not_modified(self):
        x = np.linspace(0.0, 1.0, 50)
        y = np.exp(-x)
        before = y.copy()
        unit_area(x, y)
        np.testing.assert_array_equal(y, before)

    def test_zero_field_still_raises(self):
        x = np.linspace(0.0, 1.0, 10)
        with pytest.raises(ValueError, match="cannot normalise"):
            unit_area(x, np.zeros_like(x))
