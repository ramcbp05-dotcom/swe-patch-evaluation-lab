"""Fail-to-pass tests for variance-cancellation.

Tolerances come from measurement. Relative error in the variance for a stable
algorithm grows like n*eps*kappa, kappa = mean/sigma, while the one-pass
sum-of-squares form grows like kappa**2:

    kappa     Welford    two-pass    sum-of-squares
    1e9       5e-09      1e-14       1.3e+02
    1e10      2e-07      4e-13       1.0e+00
    1e12      1.4e-07    1.5e-08     2.1e+08

so 1e-5 separates correct from broken by three orders of magnitude without
asking for accuracy float64 does not have.
"""

import numpy as np
import pytest

from fieldlab import stats

TOL = 1e-5


class TestReportedCase:
    def test_gauge_readings(self):
        readings = 1.0e8 + np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        got = stats.variance(readings)
        assert got >= 0.0, f"variance came out {got!r}; a variance cannot be negative"
        assert got == pytest.approx(0.00025, rel=TOL), (
            f"variance={got!r}, expected 0.00025"
        )

    def test_stddev_does_not_raise(self):
        readings = 1.0e8 + np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        assert stats.stddev(readings) == pytest.approx(0.015811388, rel=TOL)


class TestShiftInvariance:
    @pytest.mark.parametrize("offset", [0.0, 1e3, 1e6, 1e9, 1e12])
    def test_offset_does_not_change_the_variance(self, offset):
        a = np.random.default_rng(0).normal(0.0, 1.0, size=2000)
        base = float(np.var(a, ddof=1))
        got = stats.variance(a + offset)
        assert got == pytest.approx(base, rel=TOL), (
            f"offset {offset:g}: variance={got:.9g}, expected {base:.9g} "
            f"(kappa = {offset:.0e})"
        )

    @pytest.mark.parametrize("offset", [0.0, 1e6, 1e9, 1e12, 1e15])
    def test_never_negative(self, offset):
        a = offset + np.random.default_rng(1).normal(0.0, 1.0, size=500)
        assert stats.variance(a) >= 0.0, (
            f"offset {offset:g}: variance={stats.variance(a)!r}"
        )

    def test_extreme_condition_number(self):
        """Spread 1e-12 of the mean."""
        dev = np.random.default_rng(2).normal(0.0, 1e-3, size=4000)
        values = (1.0e9 + dev).astype(float)
        # (offset + d) - offset is exact for same-scale doubles, so this is the
        # variance of the deviations that were actually stored.
        want = float(np.var(values - 1.0e9, ddof=1))
        got = stats.variance(values)
        assert got >= 0.0, f"variance={got!r}"
        assert got == pytest.approx(want, rel=1e-4), (
            f"variance={got:.12g}, expected {want:.12g} at kappa ~ 1e12"
        )

    def test_constant_data_is_exactly_zero(self):
        for value in (1.0, 1e6, 1e9, 1e12):
            got = stats.variance([value] * 100)
            assert got == 0.0, f"constant data at {value:g} gave {got!r}, expected 0.0"


class TestEstimatorIsUnchanged:
    def test_default_is_the_sample_variance(self):
        """variance(x) must stay ddof=1: n and n-1 differ by 25% at n=5."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.variance(a) == pytest.approx(2.5, rel=1e-12), (
            f"variance={stats.variance(a)}; 2.0 means ddof became 0"
        )

    def test_ddof_zero_is_still_available(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.variance(a, ddof=0) == pytest.approx(2.0, rel=1e-12)

    @pytest.mark.parametrize("ddof", [0, 1, 2])
    def test_ddof_matches_numpy(self, ddof):
        a = np.random.default_rng(3).normal(5.0, 2.0, size=200)
        assert stats.variance(a, ddof=ddof) == pytest.approx(
            float(np.var(a, ddof=ddof)), rel=1e-9
        )

    def test_full_double_precision_on_easy_data(self):
        """A float32 shortcut shows up here: ~7 digits instead of ~15."""
        a = np.random.default_rng(4).normal(0.0, 1.0, size=1000)
        assert stats.variance(a) == pytest.approx(float(np.var(a, ddof=1)), rel=1e-12)

    def test_too_few_samples_still_raises(self):
        with pytest.raises(ValueError, match="need more than"):
            stats.variance([1.0])
