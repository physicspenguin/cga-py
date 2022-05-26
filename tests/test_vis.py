import cga_py as ga
import numpy as np


def test_cube_gen():
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 2.0, 2.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 2.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 1.0],
            [2.0, 1.0, 2.0],
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    xc = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.5, 0.0, 1.0],
            [0.0, 0.5, 0.5, 1.0],
            [0.0, 0.5, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.5, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.5, 0.0, 0.0, 1.0],
            [0.5, 0.0, 0.5, 1.0],
            [0.5, 0.0, 1.0, 1.0],
            [0.5, 0.5, 0.0, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 1.0],
            [0.5, 1.0, 0.0, 1.0],
            [0.5, 1.0, 0.5, 1.0],
            [0.5, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.5, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 0.5, 0.0, 1.0],
            [1.0, 0.5, 0.5, 1.0],
            [1.0, 0.5, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    y, yc = ga.point_cube_gen(
        np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])
    )

    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(xc, yc)


def test_point_pact():
    x = np.array([[0, 0, 0], [1, 2, 3], [4, -3, 4.6]])
    r = ga.rand_rot_poly()
    t = np.random.rand()
    y = ga.point_p_act(x, t, [r, 1])
    y1 = np.array(
        [
            ga.point_to_cartesian(ga.poly_act(t, [r, 1], ga.point(x[i])))
            for i in range(len(x))
        ]
    )

    np.testing.assert_array_equal(y, y1)