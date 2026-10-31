import numpy as np
import pytest

from fieldlab import arrays


class TestAsPoints:
    def test_1d_becomes_column_of_points(self):
        out = arrays.as_points([1.0, 2.0, 3.0])
        assert out.shape == (3, 1), "a 1-D input is n points in 1-D, not one n-D point"

    def test_2d_passes_through(self):
        out = arrays.as_points([[1.0, 2.0], [3.0, 4.0]])
        assert out.shape == (2, 2)

    def test_scalar_becomes_single_point(self):
        assert arrays.as_points(5.0).shape == (1, 1)

    def test_3d_is_rejected(self):
        with pytest.raises(ValueError, match="at most 2 dimensions"):
            arrays.as_points(np.zeros((2, 2, 2)))


class TestPairwiseDistances:
    def test_shape_is_na_by_nb(self):
        d = arrays.pairwise_distances(np.zeros((4, 2)), np.zeros((7, 2)))
        assert d.shape == (4, 7), f"expected (4, 7), got {d.shape}"

    def test_rectangular_shape_1d(self):
        d = arrays.pairwise_distances([0.0, 1.0, 2.0], [0.0, 10.0])
        assert d.shape == (3, 2)
        np.testing.assert_allclose(d, [[0.0, 10.0], [1.0, 9.0], [2.0, 8.0]])

    def test_known_2d_values(self):
        a = [[0.0, 0.0], [3.0, 4.0]]
        b = [[0.0, 0.0], [1.0, 0.0]]
        np.testing.assert_allclose(
            arrays.pairwise_distances(a, b), [[0.0, 1.0], [5.0, np.hypot(2.0, 4.0)]]
        )

    def test_self_distance_is_zero_on_the_diagonal(self, rng):
        p = rng.normal(size=(12, 3))
        d = arrays.pairwise_distances(p, p)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)

    def test_symmetric(self, rng):
        a = rng.normal(size=(5, 3))
        b = rng.normal(size=(6, 3))
        np.testing.assert_allclose(
            arrays.pairwise_distances(a, b), arrays.pairwise_distances(b, a).T
        )

    def test_matches_broadcast_reference(self, rng):
        a = rng.normal(size=(9, 4))
        b = rng.normal(size=(5, 4))
        want = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
        np.testing.assert_allclose(arrays.pairwise_distances(a, b), want, rtol=1e-12)

    def test_feature_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="feature dimension mismatch"):
            arrays.pairwise_distances(np.zeros((3, 2)), np.zeros((3, 5)))

    def test_triangle_inequality(self, rng):
        p = rng.normal(size=(8, 2))
        d = arrays.pairwise_distances(p, p)
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    assert d[i, j] <= d[i, k] + d[k, j] + 1e-9


class TestStackFields:
    def test_shape(self):
        out = arrays.stack_fields([[1, 2, 3], [4, 5, 6]])
        assert out.shape == (2, 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one field"):
            arrays.stack_fields([])

    def test_ragged_raises(self):
        with pytest.raises(ValueError, match="differing lengths"):
            arrays.stack_fields([[1, 2], [1, 2, 3]])
