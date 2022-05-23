import cga_py as ga
import numpy as np


def test_geometry_error_nomessage():
    with np.testing.assert_raises(ga.GeometryError):
        raise ga.GeometryError()
    with np.testing.assert_raises(ga.GeometryError):
        raise ga.GeometryError("test")
