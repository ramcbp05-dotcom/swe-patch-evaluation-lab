"""Fail-to-pass tests for silent-unsorted-interpolation."""

import numpy as np
import pytest

from fieldlab.interpolate import resample


class TestNonMonotonicIsRejected:
    def test_reported_case(self):
        x = np.array([0.0, 1.0, 3.0, 2.0, 4.0])
        y = np.array([0.0, 10.0, 30.0, 20.0, 40.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            resample(x, y, [2.5])

    def test_fully_reversed(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            resample([3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [2.5])

    def test_single_inversion_at_the_end(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            resample([0.0, 1.0, 2.0, 1.5], [0.0, 1.0, 2.0, 3.0], [0.5])

    def test_shuffled_long_trace(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 10.0, size=200)
        y = np.sin(x)
        with pytest.raises(ValueError, match="strictly increasing"):
            resample(x, y, np.linspace(x.min(), x.max(), 500), extrapolate="clamp")

    def test_it_raises_rather_than_sorting(self):
        """Sorting hides a failed upstream cleanup; the caller must be told."""
        x = np.array([0.0, 2.0, 1.0, 3.0])
        y = np.array([0.0, 20.0, 10.0, 30.0])
        with pytest.raises(ValueError):
            resample(x, y, [1.5])


class TestDuplicatesAreRejected:
    def test_repeated_abscissa(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            resample([0.0, 1.0, 1.0, 2.0], [0.0, 1.0, 5.0, 2.0], [1.0])

    def test_repeated_at_the_start(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            resample([0.0, 0.0, 1.0], [0.0, 9.0, 1.0], [0.5])

    def test_all_identical(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            resample([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], [2.0])


class TestSortedInputStillWorks:
    def test_interior_values(self):
        np.testing.assert_allclose(
            resample([0.0, 1.0, 2.0], [0.0, 10.0, 20.0], [0.5, 1.5]), [5.0, 15.0]
        )

    def test_result_stays_within_the_data_range(self):
        rng = np.random.default_rng(1)
        x = np.sort(rng.uniform(0.0, 10.0, size=200))
        y = np.sin(x)
        out = resample(x, y, np.linspace(x[0], x[-1], 500))
        assert out.min() >= y.min() - 1e-12, f"min {out.min()} below the data"
        assert out.max() <= y.max() + 1e-12, f"max {out.max()} above the data"

    def test_nodes_reproduced_exactly(self):
        rng = np.random.default_rng(2)
        x = np.sort(rng.uniform(0.0, 5.0, size=50))
        y = rng.normal(size=50)
        np.testing.assert_allclose(resample(x, y, x), y, rtol=1e-12)

    def test_extrapolation_modes_unchanged(self):
        with pytest.raises(ValueError, match="outside"):
            resample([0.0, 1.0], [0.0, 1.0], [1.5])
        np.testing.assert_allclose(
            resample([0.0, 1.0], [5.0, 9.0], [-3.0, 4.0], extrapolate="clamp"),
            [5.0, 9.0],
        )
        out = resample([0.0, 1.0], [5.0, 9.0], [-3.0, 0.5], extrapolate="nan")
        assert np.isnan(out[0]) and out[1] == pytest.approx(7.0), (
            f"extrapolate='nan' gave {out}, expected [nan, 7.0]"
        )

    def test_unknown_mode_still_raises(self):
        with pytest.raises(ValueError, match="unknown extrapolate mode"):
            resample([0.0, 1.0], [0.0, 1.0], [0.5], extrapolate="spline")
