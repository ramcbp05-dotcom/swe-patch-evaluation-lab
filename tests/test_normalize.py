import numpy as np
import pytest

from fieldlab import normalize


class TestUnitArea:
    def test_integral_is_one(self, gaussian_field):
        x, y = gaussian_field
        out = normalize.unit_area(x, y)
        assert np.trapezoid(out, x) == pytest.approx(1.0, rel=1e-10)

    def test_width_independence(self):
        """A narrow and a wide line must both integrate to 1."""
        x = np.linspace(-20.0, 20.0, 2001)
        for width in (0.5, 5.0):
            y = np.exp(-0.5 * (x / width) ** 2)
            out = normalize.unit_area(x, y)
            assert np.trapezoid(out, x) == pytest.approx(1.0, rel=1e-9), (
                f"width {width} did not normalise to unit area"
            )

    def test_shape_is_preserved(self, gaussian_field):
        x, y = gaussian_field
        assert normalize.unit_area(x, y).shape == y.shape

    def test_proportional_to_input(self, gaussian_field):
        x, y = gaussian_field
        a = normalize.unit_area(x, y)
        b = normalize.unit_area(x, 7.0 * y)
        np.testing.assert_allclose(a, b, rtol=1e-12), "scaling y must not change the result"

    def test_does_not_modify_input(self, gaussian_field):
        x, y = gaussian_field
        before = y.copy()
        normalize.unit_area(x, y)
        np.testing.assert_array_equal(y, before)

    def test_zero_area_raises(self):
        x = np.linspace(0.0, 1.0, 10)
        with pytest.raises(ValueError, match="cannot normalise"):
            normalize.unit_area(x, np.zeros_like(x))

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            normalize.unit_area(np.arange(5.0), np.arange(6.0))


class TestUnitPeak:
    def test_peak_is_one(self, rng):
        y = rng.normal(size=50)
        assert np.max(np.abs(normalize.unit_peak(y))) == pytest.approx(1.0)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="all-zero"):
            normalize.unit_peak(np.zeros(5))

    def test_does_not_modify_input(self, rng):
        y = rng.normal(size=20)
        before = y.copy()
        normalize.unit_peak(y)
        np.testing.assert_array_equal(y, before)


class TestZscore:
    def test_mean_and_std(self, rng):
        a = rng.normal(5.0, 3.0, size=500)
        out = normalize.zscore(a)
        assert out.mean() == pytest.approx(0.0, abs=1e-12)
        assert out.std(ddof=1) == pytest.approx(1.0, rel=1e-12)

    def test_does_not_modify_input(self, rng):
        a = rng.normal(size=100)
        before = a.copy()
        normalize.zscore(a)
        np.testing.assert_array_equal(
            a, before,
        ), "zscore must not write into the caller's array"

    def test_returns_a_distinct_array(self, rng):
        a = rng.normal(size=10)
        out = normalize.zscore(a)
        assert out is not a
        assert not np.shares_memory(out, a), (
            "the result must not be a view onto the caller's buffer"
        )

    def test_constant_raises(self):
        with pytest.raises(ValueError, match="constant field"):
            normalize.zscore(np.full(10, 2.0))

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="need more than"):
            normalize.zscore([1.0])
