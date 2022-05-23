import cga_py as ga
from cga_py import dorst_motions as dm
import numpy as np


def test_iso_scale():
    a = np.array(np.random.rand(3))
    np.testing.assert_array_almost_equal(
        dm.iso_scale(a).coeff, (ga.point(a) ^ ga.e_i).coeff
    )
    b = ga.rand_point()
    np.testing.assert_array_almost_equal(dm.iso_scale(b).coeff, (b ^ ga.e_i).coeff)


def test_transv():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    d = np.sqrt(14)
    with np.testing.assert_raises(ga.GeometryError):
        dm.transv(ga.point([1, 2, 3]), ga.plane([1, 2, 3], 2))
    x = dm.transv(a, b).coeff
    x /= x[0]
    y = (ga.point(a) ^ ga.plane(b, d)).coeff
    y /= y[0]
    np.testing.assert_array_almost_equal(x, y)
    x = dm.transv(ga.point(a), b).coeff
    x /= x[0]
    y = (ga.point(a) ^ ga.plane(b, d)).coeff
    y /= y[0]
    np.testing.assert_array_almost_equal(x, y)
    x = dm.transv(ga.point(a), ga.plane(b, d)).coeff
    x /= x[0]
    y = (ga.point(a) ^ ga.plane(b, d)).coeff
    y /= y[0]
    np.testing.assert_array_almost_equal(x, y)
    x = dm.transv(a, ga.plane(b, d)).coeff
    x /= x[0]
    y = (ga.point(a) ^ ga.plane(b, d)).coeff
    y /= y[0]
    np.testing.assert_array_almost_equal(x, y)
