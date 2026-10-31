"""Resampling with explicit out-of-range behaviour."""

from __future__ import annotations

import numpy as np

__all__ = ["resample"]


def resample(x, y, xi, extrapolate: str = "raise") -> np.ndarray:
    """Linearly interpolate ``(x, y)`` at the points ``xi``.

    ``extrapolate`` controls what happens outside ``[x[0], x[-1]]``:

    ``"raise"``  (default) raise :class:`ValueError`. Silently returning the
                 boundary value is the dangerous option: the caller gets a
                 plausible number with no indication that it was invented.
    ``"clamp"``  hold the boundary value.
    ``"nan"``    return ``nan`` outside the range.

    ``x`` must be strictly increasing. ``numpy.interp`` does not check this and
    returns meaningless values for unsorted input, so the check is made here.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    xq = np.asarray(xi, dtype=float)

    if xs.shape != ys.shape:
        raise ValueError(f"x and y must have the same length, got {xs.size} and {ys.size}")
    if xs.size < 2:
        raise ValueError("resample needs at least two samples")
    if not np.all(np.diff(xs) > 0.0):
        raise ValueError("x must be strictly increasing")
    if extrapolate not in ("raise", "clamp", "nan"):
        raise ValueError(f"unknown extrapolate mode {extrapolate!r}")

    below = xq < xs[0]
    above = xq > xs[-1]
    outside = below | above

    if extrapolate == "raise" and np.any(outside):
        bad = xq[outside]
        raise ValueError(
            f"{bad.size} query point(s) outside [{xs[0]}, {xs[-1]}]: "
            f"first offender {bad.flat[0]}"
        )

    out = np.interp(xq, xs, ys)
    if extrapolate == "nan":
        out = np.where(outside, np.nan, out)
    return out
