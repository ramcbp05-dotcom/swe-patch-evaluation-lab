import numpy as np
import pytest

from fieldlab import interpolate


class TestResample:
    def test_interior_values(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 10.0, 20.0])
        np.testing.assert_allclose(interpolate.resample(x, y, [0.5, 1.5]), [5.0, 15.0])

    def test_nodes_are_reproduced_exactly(self, rng):
        x = np.sort(rng.uniform(0.0, 10.0, size=40))
        y = rng.normal(size=40)
        np.testing.assert_allclose(interpolate.resample(x, y, x), y, rtol=1e-12)

    def test_result_stays_within_the_data_range(self, rng):
        x = np.sort(rng.uniform(0.0, 5.0, size=60))
        y = rng.normal(size=60)
        xi = np.linspace(x[0], x[-1], 500)
        out = interpolate.resample(x, y, xi)
        assert out.min() >= y.min() - 1e-12
        assert out.max() <= y.max() + 1e-12

    def test_out_of_range_raises_by_default(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="outside"):
            interpolate.resample(x, y, [1.5])

    def test_out_of_range_message_names_the_offender(self):
        with pytest.raises(ValueError, match="2.5"):
            interpolate.resample([0.0, 1.0], [0.0, 1.0], [2.5])

    def test_clamp_mode_holds_the_boundary(self):
        out = interpolate.resample([0.0, 1.0], [5.0, 9.0], [-3.0, 4.0], extrapolate="clamp")
        np.testing.assert_allclose(out, [5.0, 9.0])

    def test_nan_mode_marks_the_outside(self):
        out = interpolate.resample([0.0, 1.0], [5.0, 9.0], [-3.0, 0.5, 4.0],
                                   extrapolate="nan")
        assert np.isnan(out[0]) and np.isnan(out[2])
        assert out[1] == pytest.approx(7.0)

    def test_unsorted_x_raises(self):
        """np.interp returns silent garbage for unsorted abscissae."""
        with pytest.raises(ValueError, match="strictly increasing"):
            interpolate.resample([0.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.5])

    def test_duplicate_x_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            interpolate.resample([0.0, 1.0, 1.0], [0.0, 1.0, 2.0], [0.5])

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown extrapolate mode"):
            interpolate.resample([0.0, 1.0], [0.0, 1.0], [0.5], extrapolate="linear")

    def test_scalar_query_keeps_its_shape(self):
        out = interpolate.resample([0.0, 1.0], [0.0, 10.0], 0.25)
        assert np.ndim(out) == 0
        assert float(out) == pytest.approx(2.5)
