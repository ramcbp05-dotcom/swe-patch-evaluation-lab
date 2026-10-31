"""Numerically stable summary statistics."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["variance", "stddev", "weighted_mean", "merge_moments"]


def variance(x, ddof: int = 1) -> float:
    """Variance of ``x``, computed in a way that survives a large offset.

    The textbook one-pass form ``(sum(x**2) - sum(x)**2/n) / (n - ddof)``
    subtracts two nearly equal quantities. For data centred on 1e8 with a
    spread of 1e-2 both terms are of order 1e16 while the answer is of order
    1e-4, so every digit falls below the rounding error of the operands and the
    result can come out negative.

    Welford's recurrence accumulates squared deviations from the *running*
    mean, keeping every intermediate on the scale of the spread. Its relative
    error grows with the condition number ``mean/sigma`` rather than its
    square.
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.size
    if n <= ddof:
        raise ValueError(f"need more than {ddof} samples for ddof={ddof}, got {n}")

    mean = 0.0
    m2 = 0.0
    for count, value in enumerate(arr, start=1):
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)

    return max(m2, 0.0) / (n - ddof)


def stddev(x, ddof: int = 1) -> float:
    """Standard deviation, via :func:`variance`."""
    return math.sqrt(variance(x, ddof=ddof))


def weighted_mean(x, w) -> float:
    """Weighted arithmetic mean ``sum(w*x) / sum(w)``.

    Weights are converted to float before any division. Integer weights
    divided in integer arithmetic truncate: a weight vector of ``[1, 2]``
    would give normalised weights of ``[0, 0]``.
    """
    values = np.asarray(x, dtype=float).ravel()
    weights = np.asarray(w, dtype=float).ravel()
    if values.shape != weights.shape:
        raise ValueError(
            f"x and w must have the same length, got {values.size} and {weights.size}"
        )
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")

    total = float(weights.sum())
    if total == 0.0:
        raise ValueError("weights sum to zero")
    return float(np.dot(weights, values) / total)


def merge_moments(a: tuple[int, float, float], b: tuple[int, float, float]):
    """Merge two ``(count, mean, m2)`` accumulators.

    ``m2`` is the sum of squared deviations from that group's own mean. The
    between-group term ``delta**2 * n_a * n_b / n`` is the part that is easy to
    drop: two groups with zero internal spread but different means have
    non-zero pooled variance.
    """
    n_a, mean_a, m2_a = a
    n_b, mean_b, m2_b = b
    if n_a < 0 or n_b < 0:
        raise ValueError("counts must be non-negative")
    if n_b == 0:
        return a
    if n_a == 0:
        return b

    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    m2 = m2_a + m2_b + delta * delta * n_a * (n_b / n)
    return n, mean, m2
