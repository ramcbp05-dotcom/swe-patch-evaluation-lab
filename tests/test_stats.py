import numpy as np
import pytest

from fieldlab import stats


class TestVariance:
    def test_matches_numpy_on_well_conditioned_data(self, rng):
        a = rng.normal(3.0, 2.0, size=500)
        assert stats.variance(a) == pytest.approx(float(np.var(a, ddof=1)), rel=1e-10)

    def test_ddof_zero(self, rng):
        a = rng.normal(size=200)
        assert stats.variance(a, ddof=0) == pytest.approx(float(np.var(a, ddof=0)), rel=1e-10)

    def test_exact_small_case(self):
        assert stats.variance([1.0, 2.0, 3.0]) == pytest.approx(1.0, rel=1e-14)

    def test_offset_invariance(self, rng):
        """Variance is shift invariant; a one-pass sum-of-squares form is not."""
        a = rng.normal(0.0, 1.0, size=2000)
        base = stats.variance(a)
        for offset in (1e3, 1e6, 1e9):
            shifted = stats.variance(a + offset)
            assert shifted == pytest.approx(base, rel=1e-6), (
                f"offset {offset:g} changed the variance from {base:.9g} to "
                f"{shifted:.9g}"
            )

    def test_large_offset_small_spread(self):
        a = 1.0e8 + np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        want = float(np.var(a - 1.0e8, ddof=1))
        got = stats.variance(a)
        assert got >= 0.0, f"variance came out negative: {got!r}"
        assert got == pytest.approx(want, rel=1e-6)

    def test_never_negative(self, rng):
        for offset in (0.0, 1e6, 1e9, 1e12):
            a = offset + rng.normal(0.0, 1.0, size=500)
            assert stats.variance(a) >= 0.0, f"negative variance at offset {offset:g}"

    def test_constant_data_is_exactly_zero(self):
        assert stats.variance([1e9] * 100) == 0.0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="need more than"):
            stats.variance([1.0])


class TestStddev:
    def test_is_sqrt_of_variance(self, rng):
        a = rng.normal(size=100)
        assert stats.stddev(a) == pytest.approx(np.sqrt(stats.variance(a)), rel=1e-14)


class TestWeightedMean:
    def test_uniform_weights_give_the_plain_mean(self, rng):
        a = rng.normal(size=50)
        assert stats.weighted_mean(a, np.ones(50)) == pytest.approx(a.mean(), rel=1e-12)

    def test_integer_weights_do_not_truncate(self):
        """[1, 2] normalised in integer arithmetic would give [0, 0]."""
        got = stats.weighted_mean([10.0, 20.0], [1, 2])
        assert got == pytest.approx(50.0 / 3.0, rel=1e-12), (
            f"weighted_mean([10, 20], [1, 2]) = {got}, expected {50 / 3:.6f}"
        )

    def test_integer_weight_list_matches_float_list(self):
        assert stats.weighted_mean([1.0, 2.0, 3.0], [1, 2, 3]) == pytest.approx(
            stats.weighted_mean([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), rel=1e-14
        )

    def test_known_value(self):
        assert stats.weighted_mean([1.0, 11.0], [9.0, 1.0]) == pytest.approx(2.0)

    def test_scaling_weights_changes_nothing(self, rng):
        a = rng.normal(size=20)
        w = rng.uniform(0.1, 5.0, size=20)
        assert stats.weighted_mean(a, w) == pytest.approx(
            stats.weighted_mean(a, 100.0 * w), rel=1e-12
        )

    def test_zero_weights_raise(self):
        with pytest.raises(ValueError, match="sum to zero"):
            stats.weighted_mean([1.0, 2.0], [0.0, 0.0])

    def test_negative_weights_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            stats.weighted_mean([1.0, 2.0], [-1.0, 2.0])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            stats.weighted_mean([1.0, 2.0], [1.0])


class TestMergeMoments:
    @staticmethod
    def moments(data):
        data = np.asarray(data, dtype=float)
        mean = float(data.mean())
        return data.size, mean, float(((data - mean) ** 2).sum())

    def test_merge_matches_single_pass(self, rng):
        a = rng.normal(-5.0, 1.0, size=137)
        b = rng.normal(40.0, 9.0, size=4021)
        n, mean, m2 = stats.merge_moments(self.moments(a), self.moments(b))
        allv = np.concatenate([a, b])
        assert n == allv.size
        assert mean == pytest.approx(float(allv.mean()), rel=1e-12)
        assert m2 / (n - 1) == pytest.approx(float(np.var(allv, ddof=1)), rel=1e-10)

    def test_between_group_term_is_present(self):
        """Two zero-spread groups 10 apart have non-zero pooled variance."""
        n, mean, m2 = stats.merge_moments((100, 0.0, 0.0), (100, 10.0, 0.0))
        assert mean == pytest.approx(5.0)
        data = np.concatenate([np.zeros(100), np.full(100, 10.0)])
        assert m2 / (n - 1) == pytest.approx(float(np.var(data, ddof=1)), rel=1e-12)

    def test_unequal_sizes_weight_the_mean(self):
        _, mean, _ = stats.merge_moments((9000, 1.0, 0.0), (1000, 11.0, 0.0))
        assert mean == pytest.approx(2.0, rel=1e-12), (
            f"mean={mean}; 6.0 would mean the group means were averaged unweighted"
        )

    def test_empty_group_is_a_no_op(self):
        acc = (10, 3.0, 4.0)
        assert stats.merge_moments(acc, (0, 0.0, 0.0)) == acc
        assert stats.merge_moments((0, 0.0, 0.0), acc) == acc

    def test_order_independence(self, rng):
        a = self.moments(rng.normal(size=51))
        b = self.moments(rng.normal(10.0, 3.0, size=903))
        fwd = stats.merge_moments(a, b)
        rev = stats.merge_moments(b, a)
        assert fwd[0] == rev[0]
        assert fwd[1] == pytest.approx(rev[1], rel=1e-12)
        assert fwd[2] == pytest.approx(rev[2], rel=1e-10)
