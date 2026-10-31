"""Fail-to-pass tests for floor-division-weights."""

import numpy as np
import pytest

from fieldlab.stats import weighted_mean


class TestReportedCases:
    def test_two_integer_weights(self):
        got = weighted_mean([10.0, 20.0], [1, 2])
        assert got == pytest.approx(50.0 / 3.0, rel=1e-12), (
            f"weighted_mean([10, 20], [1, 2]) = {got}, expected {50 / 3:.12f}"
        )

    def test_uniform_weights_give_the_plain_mean(self):
        got = weighted_mean([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        assert got == pytest.approx(2.0, rel=1e-12), f"got {got}, expected 2.0"

    def test_weights_summing_to_one(self):
        got = weighted_mean([5.0, 5.0, 5.0], [0.2, 0.3, 0.5])
        assert got == pytest.approx(5.0, rel=1e-12), f"got {got}, expected 5.0"

    def test_known_value(self):
        assert weighted_mean([1.0, 11.0], [9.0, 1.0]) == pytest.approx(2.0, rel=1e-12)

    def test_single_weight_still_works(self):
        assert weighted_mean([42.0], [1.0]) == pytest.approx(42.0)


class TestNoTruncation:
    @pytest.mark.parametrize("x,w,want", [
        ([10.0, 20.0], [1, 2], 50.0 / 3.0),
        ([0.0, 1.0], [1, 1], 0.5),
        ([1.0, 2.0], [1, 3], 1.75),
        ([3.0, 7.0], [2, 5], 41.0 / 7.0),
        ([1.0, 2.0, 3.0], [1, 1, 1], 2.0),
    ])
    def test_fractional_answers_survive(self, x, w, want):
        """A truncating implementation returns the floor of these."""
        got = weighted_mean(x, w)
        assert got == pytest.approx(want, rel=1e-12), (
            f"weighted_mean({x}, {w}) = {got}, expected {want:.12f}. "
            f"A value of {int(want)}.0 means the result is being floored."
        )

    def test_answer_is_not_an_integer(self):
        got = weighted_mean([10.0, 20.0], [1, 2])
        assert got != float(int(got)), (
            f"got exactly {got}, an integer: the true answer 16.666... is not"
        )


class TestWeightConventions:
    def test_integer_and_float_weights_agree(self):
        assert weighted_mean([1.0, 2.0, 3.0], [1, 2, 3]) == pytest.approx(
            weighted_mean([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), rel=1e-14
        )

    @pytest.mark.parametrize("factor", [1e-6, 0.5, 1.0, 3.0, 1e6])
    def test_rescaling_the_weights_changes_nothing(self, factor):
        """Rounding normalised weights to 2 dp breaks this invariance."""
        rng = np.random.default_rng(4)
        x = rng.normal(10.0, 3.0, size=25)
        w = rng.uniform(0.1, 5.0, size=25)
        assert weighted_mean(x, factor * w) == pytest.approx(
            weighted_mean(x, w), rel=1e-12
        ), f"factor {factor:g} changed the weighted mean"

    def test_matches_numpy_average(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=200)
        w = rng.uniform(0.0, 10.0, size=200)
        assert weighted_mean(x, w) == pytest.approx(
            float(np.average(x, weights=w)), rel=1e-12
        )

    def test_tiny_weights_still_normalise(self):
        """Weights far below their own sum are the floor-division worst case."""
        x = [1.0, 2.0, 3.0, 4.0]
        w = [1e-8, 1e-8, 1e-8, 1e-8]
        assert weighted_mean(x, w) == pytest.approx(2.5, rel=1e-10)

    def test_one_dominant_weight(self):
        got = weighted_mean([1.0, 100.0], [1e6, 1.0])
        assert got == pytest.approx((1e6 * 1.0 + 100.0) / (1e6 + 1.0), rel=1e-12)

    def test_third_decimal_is_right(self):
        """Rounded weights are off here in the third decimal place."""
        x = [1.0, 2.0, 3.0]
        w = [1.0, 2.0, 4.0]
        want = (1.0 + 4.0 + 12.0) / 7.0
        got = weighted_mean(x, w)
        assert got == pytest.approx(want, rel=1e-12), (
            f"got {got:.12f}, expected {want:.12f}"
        )


class TestContractIsPreserved:
    def test_zero_weights_still_raise(self):
        with pytest.raises(ValueError, match="sum to zero"):
            weighted_mean([1.0, 2.0], [0.0, 0.0])

    def test_negative_weights_still_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            weighted_mean([1.0, 2.0], [-1.0, 2.0])

    def test_length_mismatch_still_raises(self):
        with pytest.raises(ValueError, match="same length"):
            weighted_mean([1.0, 2.0], [1.0])
