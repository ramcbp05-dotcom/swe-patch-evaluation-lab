"""Fail-to-pass tests for input-array-mutated."""

import numpy as np
import pytest

from fieldlab.normalize import unit_area, unit_peak, zscore


class TestInputIsNotModified:
    def test_reported_case(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        zscore(a)
        np.testing.assert_array_equal(
            a, [1.0, 2.0, 3.0, 4.0],
        ), "zscore wrote into the array it was given"

    @pytest.mark.parametrize("seed", range(6))
    def test_random_arrays_are_untouched_bitwise(self, seed):
        """Bit-exact: a restore-after-mutating fix is off by a rounding error."""
        a = np.random.default_rng(seed).normal(3.0, 2.0, size=997)
        before = a.copy()
        zscore(a)
        assert np.array_equal(a, before), (
            f"seed {seed}: input changed; max difference "
            f"{np.max(np.abs(a - before)):.3e}. A pure function must leave its "
            "argument bit-for-bit identical."
        )

    def test_read_only_input(self):
        """The decisive check: an in-place write cannot succeed here at all."""
        a = np.arange(5.0)
        a.flags.writeable = False
        out = zscore(a)
        assert out.mean() == pytest.approx(0.0, abs=1e-12)
        assert out.std(ddof=1) == pytest.approx(1.0, rel=1e-12)

    def test_view_of_a_larger_array(self):
        """The reported failure mode: scoring one column must not touch the rest."""
        runs = np.arange(40.0).reshape(10, 4)
        before = runs.copy()
        zscore(runs[:, 1])
        np.testing.assert_array_equal(
            runs, before,
        ), "scoring one column modified the parent array"

    def test_strided_view(self):
        a = np.arange(20.0)
        before = a.copy()
        zscore(a[::3])
        np.testing.assert_array_equal(a, before)

    def test_input_survives_the_error_path(self):
        """A constant field raises; the array must still be intact afterwards."""
        a = np.full(10, 7.0)
        before = a.copy()
        with pytest.raises(ValueError, match="constant field"):
            zscore(a)
        np.testing.assert_array_equal(
            a, before,
        ), "the input was left modified after zscore raised"

    def test_calling_twice_gives_the_same_answer(self):
        """If the input is mutated, the second call sees different data."""
        a = np.random.default_rng(9).normal(5.0, 2.0, size=200)
        first = zscore(a)
        second = zscore(a)
        np.testing.assert_allclose(first, second, rtol=1e-12)


class TestReturnValueDoesNotAlias:
    def test_result_is_a_distinct_object(self):
        a = np.random.default_rng(1).normal(size=50)
        out = zscore(a)
        assert out is not a, "zscore returned the very array it was given"
        assert not np.shares_memory(out, a), (
            "the returned array shares memory with the input, so writing to the "
            "result would corrupt the caller's data"
        )

    def test_writing_to_the_result_is_safe(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        before = a.copy()
        out = zscore(a)
        out[:] = 999.0
        np.testing.assert_array_equal(a, before)


class TestSiblingFunctionsArePureToo:
    def test_unit_peak_does_not_modify_input(self):
        y = np.random.default_rng(2).normal(size=40)
        before = y.copy()
        unit_peak(y)
        assert np.array_equal(y, before)

    def test_unit_area_does_not_modify_input(self):
        x = np.linspace(0.0, 1.0, 60)
        y = np.exp(-x)
        before = y.copy()
        unit_area(x, y)
        assert np.array_equal(y, before)

    def test_unit_peak_read_only(self):
        y = np.arange(1.0, 6.0)
        y.flags.writeable = False
        assert np.max(np.abs(unit_peak(y))) == pytest.approx(1.0), (
            "unit_peak failed on a read-only array"
        )


class TestCorrectnessIsUnchanged:
    def test_zscore_statistics(self):
        a = np.random.default_rng(3).normal(5.0, 3.0, size=500)
        out = zscore(a)
        assert out.mean() == pytest.approx(0.0, abs=1e-12)
        assert out.std(ddof=1) == pytest.approx(1.0, rel=1e-12)

    def test_list_input_still_works(self):
        out = zscore([1.0, 2.0, 3.0, 4.0])
        assert out.mean() == pytest.approx(0.0, abs=1e-12)

    def test_too_few_samples_still_raises(self):
        with pytest.raises(ValueError, match="need more than"):
            zscore([1.0])
