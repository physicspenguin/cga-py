import cga_py as ga
import numpy as np


def test_point_array_error():
    with np.testing.assert_raises(ValueError):
        ga.point(np.random.rand(2))
    with np.testing.assert_raises(ValueError):
        ga.point(np.random.rand(4))
    with np.testing.assert_raises(ValueError):
        ga.point(np.random.rand(1))
    with np.testing.assert_raises(ValueError):
        ga.point(np.random.rand(0))


def test_normalize_point_error():
    a = ga.rand_point()
    b = a + np.random.rand()
    c = a + ga.cga_object(np.append(np.zeros(6), np.random.rand(32 - 6)))
    with np.testing.assert_raises(ValueError):
        ga.normalize_point(b)
    with np.testing.assert_raises(ValueError):
        ga.normalize_point(c)


def test_sphere_array_error():
    with np.testing.assert_raises(ValueError):
        ga.sphere(np.random.rand(2), 5)
    with np.testing.assert_raises(ValueError):
        ga.sphere(np.random.rand(4), 6)
    with np.testing.assert_raises(ValueError):
        ga.sphere(np.random.rand(1), 7)
    with np.testing.assert_raises(ValueError):
        ga.sphere(np.random.rand(0), 8)
    with np.testing.assert_raises(TypeError):
        ga.sphere(np.random.rand(3))
    with np.testing.assert_raises(TypeError):
        ga.sphere(np.random.rand(2))
    with np.testing.assert_raises(TypeError):
        ga.sphere(np.random.rand(4))
    with np.testing.assert_raises(TypeError):
        ga.sphere(np.random.rand(1))
    with np.testing.assert_raises(TypeError):
        ga.sphere(np.random.rand(0))


def test_normalize_sphere_error():
    a = ga.rand_sphere()
    b = a + np.random.rand()
    c = a + ga.cga_object(np.append(np.zeros(6), np.random.rand(32 - 6)))
    with np.testing.assert_raises(ValueError):
        ga.normalize_sphere(b)
    with np.testing.assert_raises(ValueError):
        ga.normalize_sphere(c)


def test_plane_array_error():
    with np.testing.assert_raises(ValueError):
        ga.plane(np.random.rand(2), 5)
    with np.testing.assert_raises(ValueError):
        ga.plane(np.random.rand(4), 6)
    with np.testing.assert_raises(ValueError):
        ga.plane(np.random.rand(1), 7)
    with np.testing.assert_raises(ValueError):
        ga.plane(np.random.rand(0), 8)
    with np.testing.assert_raises(TypeError):
        ga.plane(np.random.rand(3))
    with np.testing.assert_raises(TypeError):
        ga.plane(np.random.rand(2))
    with np.testing.assert_raises(TypeError):
        ga.plane(np.random.rand(4))
    with np.testing.assert_raises(TypeError):
        ga.plane(np.random.rand(1))
    with np.testing.assert_raises(TypeError):
        ga.plane(np.random.rand(0))


def test_normalize_plane_error():
    a = ga.rand_plane()
    b = a + np.random.rand()
    c = a + ga.cga_object(np.append(np.zeros(5), np.random.rand(32 - 5)))
    with np.testing.assert_raises(ValueError):
        ga.normalize_plane(b)
    with np.testing.assert_raises(ValueError):
        ga.normalize_plane(c)
