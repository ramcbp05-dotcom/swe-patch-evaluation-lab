import numpy as np
import pytest

from fieldlab import grids


class TestNeighbourIndices:
    def test_periodic_wraps_at_both_ends(self):
        assert grids.neighbour_indices(0, 5, "periodic") == (4, 1)
        assert grids.neighbour_indices(4, 5, "periodic") == (3, 0)

    def test_periodic_interior(self):
        assert grids.neighbour_indices(2, 5, "periodic") == (1, 3)

    @pytest.mark.parametrize("n", [2, 3, 7, 16, 101])
    def test_periodic_right_neighbour_stays_in_range(self, n):
        """`i + 1 % n` is `i + (1 % n)`. It runs off the end for every n > 1."""
        for i in range(n):
            left, right = grids.neighbour_indices(i, n, "periodic")
            assert 0 <= right < n, f"n={n} i={i}: right neighbour {right} out of range"
            assert 0 <= left < n, f"n={n} i={i}: left neighbour {left} out of range"

    def test_periodic_is_a_cycle(self):
        n = 9
        i = 0
        for _ in range(n):
            i = grids.neighbour_indices(i, n, "periodic")[1]
        assert i == 0, "stepping right n times must return to the start"

    def test_neumann_reflects(self):
        assert grids.neighbour_indices(0, 5, "neumann") == (0, 1)
        assert grids.neighbour_indices(4, 5, "neumann") == (3, 4)

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            grids.neighbour_indices(5, 5)

    def test_unknown_bc_raises(self):
        with pytest.raises(ValueError, match="unknown boundary condition"):
            grids.neighbour_indices(0, 5, "sponge")


class TestLaplacian1d:
    def test_linear_field_has_zero_interior_laplacian(self):
        u = np.arange(10.0)
        out = grids.laplacian_1d(u, dx=1.0, bc="neumann")
        np.testing.assert_allclose(out[1:-1], 0.0, atol=1e-12)

    def test_quadratic_field_recovers_the_second_derivative(self):
        x = np.linspace(0.0, 1.0, 101)
        dx = x[1] - x[0]
        out = grids.laplacian_1d(x ** 2, dx=dx, bc="neumann")
        np.testing.assert_allclose(out[1:-1], 2.0, rtol=1e-9)

    def test_neumann_conserves_total_mass(self, rng):
        """Zero flux through the walls means the stencil must sum to zero."""
        u = rng.normal(size=40)
        out = grids.laplacian_1d(u, dx=0.25, bc="neumann")
        assert abs(out.sum()) < 1e-9 * max(1.0, np.abs(u).sum()), (
            f"sum of the Neumann Laplacian is {out.sum():.3e}, expected 0: a "
            "non-zero sum means material leaks through the boundary"
        )

    def test_neumann_boundary_stencil_is_one_sided(self):
        u = np.array([1.0, 2.0, 4.0, 7.0])
        out = grids.laplacian_1d(u, dx=1.0, bc="neumann")
        assert out[0] == pytest.approx(1.0), "cell 0 must be (u1 - u0)/dx**2"
        assert out[-1] == pytest.approx(-3.0), "last cell must be (u[-2] - u[-1])/dx**2"

    def test_periodic_conserves_total_mass(self, rng):
        u = rng.normal(size=32)
        out = grids.laplacian_1d(u, dx=1.0, bc="periodic")
        assert abs(out.sum()) < 1e-9

    def test_periodic_eigenvector(self):
        """A single Fourier mode is an eigenvector with a known eigenvalue."""
        n = 64
        k = 3
        i = np.arange(n)
        u = np.cos(2 * np.pi * k * i / n)
        out = grids.laplacian_1d(u, dx=1.0, bc="periodic")
        eig = 2.0 * (np.cos(2 * np.pi * k / n) - 1.0)
        np.testing.assert_allclose(out, eig * u, rtol=1e-10, atol=1e-12)

    def test_dx_scaling(self, rng):
        u = rng.normal(size=20)
        a = grids.laplacian_1d(u, dx=1.0, bc="periodic")
        b = grids.laplacian_1d(u, dx=2.0, bc="periodic")
        np.testing.assert_allclose(b, a / 4.0, rtol=1e-12)

    def test_too_small_grid_raises(self):
        with pytest.raises(ValueError, match="at least 2 cells"):
            grids.laplacian_1d([1.0])

    def test_bad_dx_raises(self):
        with pytest.raises(ValueError, match="dx must be positive"):
            grids.laplacian_1d([1.0, 2.0, 3.0], dx=0.0)


class TestGradient1d:
    def test_linear_field(self):
        x = np.linspace(0.0, 2.0, 50)
        out = grids.gradient_1d(3.0 * x, dx=x[1] - x[0], bc="neumann")
        np.testing.assert_allclose(out, 3.0, rtol=1e-9)

    def test_periodic_mode(self):
        n = 64
        i = np.arange(n)
        u = np.sin(2 * np.pi * i / n)
        out = grids.gradient_1d(u, dx=1.0, bc="periodic")
        assert np.all(np.isfinite(out))
        assert abs(out.sum()) < 1e-10
