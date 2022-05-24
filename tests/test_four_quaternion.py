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
    a = ga.quat_to_rotor(q0, q1, q2, q3)
    np.testing.assert_array_almost_equal(r.coeff, a.coeff)
    q0 = ga.quat_to_arr(q0)
    q1 = ga.quat_to_arr(q1)
    q2 = ga.quat_to_arr(q2)
    q3 = ga.quat_to_arr(q3)
    a = ga.quat_to_rotor(q0, q1, q2, q3)
    np.testing.assert_array_almost_equal(r.coeff, a.coeff)


def test_vectorial():
    q = np.random.rand(4)
    p1 = ga.vectorial(q)
    np.testing.assert_array_almost_equal(p1, q[1:-1])
    q2 = q
    q2[0] = 0
    np.testing.assert_array_almost_equal(
        ga.vectorial(ga.arr_to_quat(q)).coeff, ga.arr_to_quat(q2).coeff
    )


def test_scalar():
    q = np.random.rand(4)
    np.testing.assert_array_almost_equal(
        ga.scalar(ga.arr_to_quat(q)).coeff, ga.arr_to_quat([q[0], 0, 0, 0]).coeff
    )
    np.testing.assert_almost_equal(q[0], ga.scalar(q))
