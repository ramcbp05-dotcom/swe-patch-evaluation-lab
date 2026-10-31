"""Fail-to-pass tests for neumann-boundary-flux."""

import numpy as np
import pytest

from fieldlab.grids import laplacian_1d


class TestConservation:
    @pytest.mark.parametrize("n", [2, 3, 4, 8, 17, 64, 129])
    def test_row_sums_are_zero(self, n):
        """Zero flux at both walls means every stencil term cancels."""
        rng = np.random.default_rng(n)
        u = rng.normal(size=n)
        out = laplacian_1d(u, dx=1.0, bc="neumann")
        scale = max(1.0, float(np.abs(out).sum()))
        assert abs(out.sum()) < 1e-10 * scale, (
            f"n={n}: sum of the Neumann Laplacian is {out.sum():.6e}, expected 0. "
            "A non-zero sum is material crossing a wall that has no flux."
        )

    @pytest.mark.parametrize("dx", [0.1, 1.0, 3.5])
    def test_conservation_is_independent_of_dx(self, dx):
        u = np.random.default_rng(3).normal(size=33)
        out = laplacian_1d(u, dx=dx, bc="neumann")
        assert abs(out.sum()) < 1e-9 * max(1.0, float(np.abs(out).sum()))

    def test_reported_diffusion_run_conserves_mass(self):
        u = np.zeros(64)
        u[30:34] = 1.0
        total0 = u.sum()
        for _ in range(4000):
            u = u + 0.2 * laplacian_1d(u, dx=1.0, bc="neumann")
        assert u.sum() == pytest.approx(total0, rel=1e-9), (
            f"total went from {total0} to {u.sum()} in a closed box"
        )

    def test_operator_matrix_rows_sum_to_zero(self):
        """Build the matrix column by column and check it directly."""
        n = 12
        matrix = np.column_stack([
            laplacian_1d(np.eye(n)[:, j], dx=1.0, bc="neumann") for j in range(n)
        ])
        np.testing.assert_allclose(matrix.sum(axis=1), 0.0, atol=1e-12)

    def test_operator_is_symmetric(self):
        """A self-adjoint operator with these boundary conditions is symmetric."""
        n = 10
        matrix = np.column_stack([
            laplacian_1d(np.eye(n)[:, j], dx=1.0, bc="neumann") for j in range(n)
        ])
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)


class TestEdgeStencil:
    def test_exact_values_from_the_issue(self):
        u = np.array([1.0, 2.0, 4.0, 7.0])
        out = laplacian_1d(u, dx=1.0, bc="neumann")
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0, -3.0], rtol=1e-12)

    def test_boundary_cell_is_a_one_sided_difference(self):
        u = np.array([0.0, 5.0, 5.0, 5.0, 2.0])
        out = laplacian_1d(u, dx=1.0, bc="neumann")
        assert out[0] == pytest.approx(5.0), (
            f"out[0] = {out[0]}, expected (u[1] - u[0])/dx**2 = 5.0"
        )
        assert out[-1] == pytest.approx(3.0), (
            f"out[-1] = {out[-1]}, expected (u[-2] - u[-1])/dx**2 = 3.0"
        )

    def test_constant_field_gives_zero_everywhere(self):
        out = laplacian_1d(np.full(20, 3.7), dx=0.5, bc="neumann")
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_linear_field_is_zero_in_the_interior(self):
        """A linear ramp has zero second derivative inside the domain."""
        out = laplacian_1d(np.arange(12.0), dx=1.0, bc="neumann")
        np.testing.assert_allclose(out[1:-1], 0.0, atol=1e-12)

    def test_linear_ramp_edges_are_plus_and_minus_one(self):
        """The ramp's edge values follow from the zero-flux ghost cell."""
        out = laplacian_1d(np.arange(12.0), dx=1.0, bc="neumann")
        assert out[0] == pytest.approx(1.0), f"out[0] = {out[0]}, expected 1.0"
        assert out[-1] == pytest.approx(-1.0), f"out[-1] = {out[-1]}, expected -1.0"

    def test_interior_matches_the_analytic_second_derivative(self):
        x = np.linspace(0.0, 1.0, 201)
        dx = x[1] - x[0]
        out = laplacian_1d(x ** 2, dx=dx, bc="neumann")
        np.testing.assert_allclose(out[1:-1], 2.0, rtol=1e-8)

    def test_dx_scaling_at_the_boundary(self):
        u = np.array([0.0, 1.0, 3.0, 6.0])
        a = laplacian_1d(u, dx=1.0, bc="neumann")
        b = laplacian_1d(u, dx=2.0, bc="neumann")
        np.testing.assert_allclose(b, a / 4.0, rtol=1e-12)


class TestOtherBoundaryConditionsUntouched:
    def test_periodic_still_conserves(self):
        u = np.random.default_rng(5).normal(size=32)
        assert abs(laplacian_1d(u, bc="periodic").sum()) < 1e-10

    def test_dirichlet_unchanged(self):
        u = np.array([1.0, 2.0, 3.0])
        out = laplacian_1d(u, dx=1.0, bc="dirichlet")
        np.testing.assert_allclose(out, [0.0, 0.0, -4.0], rtol=1e-12)
