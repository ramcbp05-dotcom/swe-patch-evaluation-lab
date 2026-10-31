"""Shape handling and pairwise geometry."""

from __future__ import annotations

import numpy as np

__all__ = ["as_points", "pairwise_distances", "stack_fields"]


def as_points(a) -> np.ndarray:
    """Coerce input to an ``(n, d)`` float array of points.

    A 1-D input is read as ``n`` points in one dimension, so ``[1, 2, 3]``
    becomes shape ``(3, 1)`` rather than a single 3-D point. That convention is
    what makes :func:`pairwise_distances` behave the same for 1-D and 2-D data.
    """
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"expected at most 2 dimensions, got shape {arr.shape}")


def pairwise_distances(a, b) -> np.ndarray:
    """Euclidean distance between every point of ``a`` and every point of ``b``.

    Returns an ``(n_a, n_b)`` array. ``a`` and ``b`` must agree in their
    trailing (feature) dimension.
    """
    pa = as_points(a)
    pb = as_points(b)
    if pa.shape[1] != pb.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: a has {pa.shape[1]}, b has {pb.shape[1]}"
        )

    # (n_a, 1, d) - (1, n_b, d) -> (n_a, n_b, d); the inserted axes are what make
    # this an outer operation rather than an elementwise one.
    diff = pa[:, None, :] - pb[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def stack_fields(fields) -> np.ndarray:
    """Stack equal-length 1-D fields into an ``(n_fields, n_samples)`` array."""
    if not len(fields):
        raise ValueError("stack_fields requires at least one field")
    arrays = [np.asarray(f, dtype=float).ravel() for f in fields]
    lengths = {a.size for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"fields have differing lengths: {sorted(lengths)}")
    return np.stack(arrays, axis=0)
