import cga_py as ga
import numpy as np


def test_poly_act():
    a = ga.cga_object(np.random.rand(32))
    x0 = ga.cga_object(np.random.rand(32))
    x1 = ga.cga_object(np.random.rand(32))
    x2 = ga.cga_object(np.random.rand(32))
    x3 = ga.cga_object(np.random.rand(32))
    x4 = ga.cga_object(np.random.rand(32))
    x5 = ga.cga_object(np.random.rand(32))
    t = np.random.rand()
    p = ga.poly_act(t, [x0, x1, x2, x3, x4, x5], a)
    pp = x0 + x1 * t + x2 * (t**2) + x3 * (t**3) + x4 * (t**4) + x5 * (t**5)
    b = ga.act(pp, a)
    np.testing.assert_array_almost_equal(p.coeff, b.coeff)
