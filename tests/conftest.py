"""Shared fixtures for the fieldlab test suite."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Fixed-seed generator: the library suite is the regression baseline, so it
    must give the same verdict on every run."""
    return np.random.default_rng(20260301)


@pytest.fixture
def gaussian_field(rng):
    """A non-uniform grid carrying a Gaussian line on a flat continuum."""
    x = np.sort(rng.uniform(0.0, 10.0, size=300))
    y = 2.0 + 5.0 * np.exp(-0.5 * ((x - 4.0) / 0.8) ** 2)
    return x, y
