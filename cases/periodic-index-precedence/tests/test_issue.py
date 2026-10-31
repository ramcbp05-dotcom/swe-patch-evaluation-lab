"""Fail-to-pass tests for periodic-index-precedence."""

import numpy as np
import pytest

from fieldlab.grids import neighbour_indices


class TestIndicesStayInRange:
    def test_reported_case(self):
        assert neighbour_indices(7, 8, "periodic") == (6, 0), (
            f"got {neighbour_indices(7, 8, 'periodic')}, expected (6, 0)"
        )

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 17, 64, 101])
    def test_every_cell_of_every_grid(self, n):
        for i in range(n):
            left, right = neighbour_indices(i, n, "periodic")
            assert 0 <= left < n, f"n={n} i={i}: left neighbour {left} out of range"
            assert 0 <= right < n, f"n={n} i={i}: right neighbour {right} out of range"

    @pytest.mark.parametrize("n", [2, 3, 8, 101])
    def test_indexing_with_the_result_does_not_raise(self, n):
        u = np.arange(float(n))
        for i in range(n):
            left, right = neighbour_indices(i, n, "periodic")
            u[left], u[right]  # must not raise


class TestItIsActuallyPeriodic:
    @pytest.mark.parametrize("n", [2, 3, 5, 9, 32])
    def test_stepping_right_n_times_returns_to_start(self, n):
        """Clamping keeps the indices in range but is not a cycle."""
        for start in range(n):
            i = start
            for _ in range(n):
                i = neighbour_indices(i, n, "periodic")[1]
            assert i == start, (
                f"n={n}: starting at {start}, stepping right {n} times landed on {i}"
            )

    @pytest.mark.parametrize("n", [2, 3, 5, 9, 32])
    def test_stepping_left_n_times_returns_to_start(self, n):
        for start in range(n):
            i = start
            for _ in range(n):
                i = neighbour_indices(i, n, "periodic")[0]
            assert i == start, (
                f"n={n}: starting at {start}, stepping left {n} times landed on {i}"
            )

    def test_wrap_at_the_left_end(self):
        """Cell 0's left neighbour is the last cell, not cell 1."""
        assert neighbour_indices(0, 8, "periodic")[0] == 7, (
            f"left neighbour of cell 0 is "
            f"{neighbour_indices(0, 8, 'periodic')[0]}, expected 7"
        )

    def test_wrap_at_the_right_end(self):
        assert neighbour_indices(7, 8, "periodic")[1] == 0

    @pytest.mark.parametrize("n", [3, 4, 7, 16])
    def test_every_cell_is_someone_elses_neighbour_exactly_once(self, n):
        """On a ring, the right-neighbour map is a permutation of the cells."""
        rights = [neighbour_indices(i, n, "periodic")[1] for i in range(n)]
        assert sorted(rights) == list(range(n)), (
            f"n={n}: right-neighbour map is {rights}, which is not a permutation. "
            "Some cell is reachable twice and another not at all"
        )

    @pytest.mark.parametrize("n", [3, 4, 7, 16])
    def test_left_and_right_are_mutually_inverse(self, n):
        for i in range(n):
            right = neighbour_indices(i, n, "periodic")[1]
            assert neighbour_indices(right, n, "periodic")[0] == i, (
                f"n={n}: stepping right from {i} to {right} then left does not return"
            )

    def test_interior_is_unaffected(self):
        assert neighbour_indices(4, 9, "periodic") == (3, 5)


class TestOtherBoundaryConditionsUntouched:
    def test_neumann_reflects(self):
        assert neighbour_indices(0, 5, "neumann") == (0, 1)
        assert neighbour_indices(4, 5, "neumann") == (3, 4)

    def test_out_of_range_index_still_raises(self):
        with pytest.raises(IndexError):
            neighbour_indices(5, 5, "periodic")

    def test_unknown_bc_still_raises(self):
        with pytest.raises(ValueError, match="unknown boundary condition"):
            neighbour_indices(0, 5, "absorbing")
