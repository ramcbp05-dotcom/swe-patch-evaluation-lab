"""Finite differences on a uniform 1-D grid, with boundary conditions."""

from __future__ import annotations

import numpy as np

__all__ = ["BOUNDARY_CONDITIONS", "neighbour_indices", "laplacian_1d", "gradient_1d"]

BOUNDARY_CONDITIONS = ("periodic", "neumann", "dirichlet")


def neighbour_indices(i: int, n: int, bc: str = "periodic") -> tuple[int, int]:
    """Indices of the left and right neighbours of cell ``i`` in a grid of ``n``.

    ``periodic`` wraps; ``neumann`` reflects (the ghost cell mirrors the
    boundary cell, giving zero gradient); ``dirichlet`` clamps and the caller is
    expected to overwrite the boundary values afterwards.
    """
    if n < 1:
        raise ValueError(f"grid size must be positive, got {n}")
    if not 0 <= i < n:
        raise IndexError(f"index {i} out of range for grid of size {n}")
    if bc not in BOUNDARY_CONDITIONS:
        raise ValueError(f"unknown boundary condition {bc!r}; expected {BOUNDARY_CONDITIONS}")

    if bc == "periodic":
        # The parentheses matter: % binds tighter than +, so `i + 1 % n` is
        # `i + (1 % n)`, which is `i + 1` for every n > 1 and runs off the end.
        return (i - 1) % n, (i + 1) % n
    if bc == "neumann":
        return max(i - 1, 0), min(i + 1, n - 1)
    return max(i - 1, 0), min(i + 1, n - 1)


def laplacian_1d(u, dx: float = 1.0, bc: str = "periodic") -> np.ndarray:
    """Second difference of ``u`` on a uniform grid of spacing ``dx``.

    With ``bc="neumann"`` the discrete operator conserves the sum of ``u``:
    the ghost cell outside each boundary equals the boundary cell, so the
    stencil at cell 0 is ``(u[1] - u[0]) / dx**2``: a *one-sided* difference,
    not the interior three-point stencil. Getting this wrong leaks or injects
    material at the edges, which only shows up as a slow drift.
    """
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    if n < 2:
        raise ValueError(f"laplacian_1d needs at least 2 cells, got {n}")
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx}")
    if bc not in BOUNDARY_CONDITIONS:
        raise ValueError(f"unknown boundary condition {bc!r}; expected {BOUNDARY_CONDITIONS}")

    out = np.empty_like(arr)
    inv = 1.0 / (dx * dx)

    out[1:-1] = (arr[2:] - 2.0 * arr[1:-1] + arr[:-2]) * inv

    if bc == "periodic":
        out[0] = (arr[1] - 2.0 * arr[0] + arr[-1]) * inv
        out[-1] = (arr[0] - 2.0 * arr[-1] + arr[-2]) * inv
    elif bc == "neumann":
        # ghost u[-1] = u[0] and u[n] = u[n-1]  =>  zero flux through the wall
        out[0] = (arr[1] - arr[0]) * inv
        out[-1] = (arr[-2] - arr[-1]) * inv
    else:  # dirichlet: ghost cells are zero
        out[0] = (arr[1] - 2.0 * arr[0]) * inv
        out[-1] = (-2.0 * arr[-1] + arr[-2]) * inv

    return out


def gradient_1d(u, dx: float = 1.0, bc: str = "periodic") -> np.ndarray:
    """Centred first difference, one-sided at a non-periodic boundary."""
    arr = np.asarray(u, dtype=float).ravel()
    n = arr.size
    if n < 2:
        raise ValueError(f"gradient_1d needs at least 2 cells, got {n}")
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx}")

    out = np.empty_like(arr)
    out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dx)
    if bc == "periodic":
        out[0] = (arr[1] - arr[-1]) / (2.0 * dx)
        out[-1] = (arr[0] - arr[-2]) / (2.0 * dx)
    else:
        out[0] = (arr[1] - arr[0]) / dx
        out[-1] = (arr[-1] - arr[-2]) / dx
    return out
