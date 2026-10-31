"""Fail-to-pass tests for shape-mismatch-pairwise-distance."""

import numpy as np
import pytest

from fieldlab.arrays import pairwise_distances


def reference(a, b):
    """Independent implementation, used only to decide what is correct."""
    a = np.atleast_2d(np.asarray(a, dtype=float).T).T if np.ndim(a) == 1 else np.asarray(a, float)
    b = np.atleast_2d(np.asarray(b, dtype=float).T).T if np.ndim(b) == 1 else np.asarray(b, float)
    a = a.reshape(-1, 1) if a.ndim == 1 else a
    b = b.reshape(-1, 1) if b.ndim == 1 else b
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))


class TestRectangularInputs:
    def test_reported_case_does_not_raise(self):
        sensors = np.random.default_rng(0).normal(size=(5, 3))
        targets = np.random.default_rng(1).normal(size=(7, 3))
        out = pairwise_distances(sensors, targets)
        assert out.shape == (5, 7), f"expected (5, 7), got {out.shape}"

    def test_rectangular_values_match_the_reference(self):
        sensors = np.random.default_rng(0).normal(size=(5, 3))
        targets = np.random.default_rng(1).normal(size=(7, 3))
        np.testing.assert_allclose(
            pairwise_distances(sensors, targets), reference(sensors, targets), rtol=1e-12
        )

    def test_1d_rectangular_case(self):
        out = pairwise_distances([0.0, 1.0, 2.0], [0.0, 10.0])
        assert out.shape == (3, 2), f"expected (3, 2), got {out.shape}"
        np.testing.assert_allclose(out, [[0.0, 10.0], [1.0, 9.0], [2.0, 8.0]])

    @pytest.mark.parametrize("n_a,n_b,d", [(1, 1, 1), (2, 9, 1), (9, 2, 4),
                                           (3, 3, 7), (12, 5, 2), (4, 4, 4)])
    def test_shape_grid(self, n_a, n_b, d):
        rng = np.random.default_rng(n_a * 100 + n_b * 10 + d)
        a = rng.normal(size=(n_a, d))
        b = rng.normal(size=(n_b, d))
        assert pairwise_distances(a, b).shape == (n_a, n_b)


class TestDistancesAreNotSquared:
    def test_known_right_triangle(self):
        """(3, 4) is 5 away from the origin: 25 would mean a missing sqrt."""
        out = pairwise_distances([[0.0, 0.0]], [[3.0, 4.0]])
        assert out[0, 0] == pytest.approx(5.0), (
            f"distance came out {out[0, 0]}; 25.0 means the square root is missing"
        )

    def test_unit_step_is_one(self):
        out = pairwise_distances([[0.0]], [[1.0]])
        assert out[0, 0] == pytest.approx(1.0)

    def test_scaling_is_linear_not_quadratic(self):
        """Doubling the separation must double the distance."""
        near = pairwise_distances([[0.0, 0.0]], [[1.0, 1.0]])[0, 0]
        far = pairwise_distances([[0.0, 0.0]], [[2.0, 2.0]])[0, 0]
        assert far / near == pytest.approx(2.0, rel=1e-12), (
            f"ratio {far / near:.3f}; 4.0 means squared distances are being returned"
        )


class TestContractIsPreserved:
    def test_feature_mismatch_still_raises(self):
        with pytest.raises(ValueError, match="feature dimension mismatch"):
            pairwise_distances(np.zeros((3, 2)), np.zeros((4, 5)))

    def test_square_case_keeps_working(self):
        p = np.random.default_rng(7).normal(size=(6, 3))
        out = pairwise_distances(p, p)
        assert out.shape == (6, 6)
        np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-12)
        np.testing.assert_allclose(out, out.T, rtol=1e-12)
