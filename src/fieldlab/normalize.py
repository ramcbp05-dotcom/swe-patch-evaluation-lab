"""Normalisations that each preserve a stated invariant."""

from __future__ import annotations

import numpy as np

__all__ = ["unit_area", "unit_peak", "zscore"]


def unit_area(x, y) -> np.ndarray:
    """Scale ``y`` so that ``trapezoid(y, x) == 1``.

    This is the normalisation to use when areas must be comparable between
    samples: unlike peak normalisation it does not depend on the width of a
    feature. ``x`` may be non-uniform.

    The input is never modified; a new array is returned.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must have the same shape, got {xs.shape} and {ys.shape}")
    if xs.size < 2:
        raise ValueError("unit_area needs at least two samples")

    area = float(np.trapezoid(ys, xs))
    if not np.isfinite(area) or area == 0.0:
        raise ValueError(f"integrated area is {area!r}; cannot normalise to unit area")
    return ys / area


def unit_peak(y) -> np.ndarray:
    """Scale ``y`` so that ``max(abs(y)) == 1``. Returns a new array."""
    ys = np.asarray(y, dtype=float)
    peak = float(np.max(np.abs(ys))) if ys.size else 0.0
    if peak == 0.0:
        raise ValueError("cannot peak-normalise an all-zero field")
    return ys / peak


def zscore(a, ddof: int = 1) -> np.ndarray:
    """Centre and scale to zero mean and unit standard deviation.

    The input is copied, so a caller's array is never modified in place: the
    functions in this module are pure, and callers rely on that when they
    normalise a view of a larger dataset.
    """
    arr = np.asarray(a, dtype=float)
    if arr.size <= ddof:
        raise ValueError(f"need more than {ddof} samples for ddof={ddof}")
    centred = arr - arr.mean()
    scale = float(centred.std(ddof=ddof))
    if scale == 0.0:
        raise ValueError("cannot z-score a constant field")
    return centred / scale
