import cga_py as ga
import numpy as np


def test_arr_to_quat():
    x = np.random.rand(4)
    np.testing.assert_array_almost_equal(
        ga.arr_to_quat(x).coeff,
        (x[0] + x[1] * ga.q_i + x[2] * ga.q_j + x[3] * ga.q_k).coeff,
    )


def test_quat_to_arr():
    x = np.random.rand(4)
    np.testing.assert_array_almost_equal(x, ga.quat_to_arr(ga.arr_to_quat(x)))


def test_rotors():
    r = ga.rand_rotor()
    q0, q1, q2, q3 = ga.rotor_to_quat(r)
    q0 = ga.quat_to_arr(q0)
    q1 = ga.quat_to_arr(q1)
    q2 = ga.quat_to_arr(q2)
    q3 = ga.quat_to_arr(q3)
    a = ga.quat_to_rotor(q0, q1, q2, q3)
    np.testing.assert_array_almost_equal(r.coeff, a.coeff)
