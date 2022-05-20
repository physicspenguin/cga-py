import cga_py as cg
import numpy as np


def test_rand_rational():
    i = np.random.randint(2000)
    assert abs(cg.rand_rational(i)) < i + 1


def test_point_generation():
    i = np.random.randint(2000)
    np.testing.assert_no_warnings(cg.point_to_cartesian, cg.rand_point(i))


def test_plane_generation():
    i = np.random.randint(2000)
    np.testing.assert_no_warnings(cg.plane_to_cartesian, cg.rand_plane(i))


def test_sphere_generation():
    i = np.random.randint(2000)
    np.testing.assert_no_warnings(cg.sphere_to_cartesian, cg.rand_sphere(i))


def test_rand_rotor():
    i = np.random.randint(2000)
    np.testing.assert_array_almost_equal(cg.study_var(cg.rand_rotor(i)), np.zeros(10))


def test_rand_rotor_poly():
    i = np.random.randint(2000)
    np.testing.assert_array_almost_equal(
        cg.study_var(cg.rand_rot_poly(i)), np.zeros(10)
    )


def test_rand_zero():
    np.testing.assert_almost_equal(cg.null_quadric(cg.rand_zero()), 0)
