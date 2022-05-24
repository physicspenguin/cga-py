import cga_py as ga
import numpy as np


def test_act():
    r = ga.cga_object(np.random.rand(32))
    a = ga.cga_object(np.random.rand(32))
    np.testing.assert_array_almost_equal(ga.act(r, a).coeff, (r * a * (~r)).coeff)


def test_com():
    r = ga.cga_object(np.random.rand(32))
    a = ga.cga_object(np.random.rand(32))
    np.testing.assert_array_almost_equal(
        ga.com(r, a).coeff, (0.5 * (r * a - a * r)).coeff
    )


def test_anti_com():
    r = ga.cga_object(np.random.rand(32))
    a = ga.cga_object(np.random.rand(32))
    np.testing.assert_array_almost_equal(
        ga.anti_com(r, a).coeff, (0.5 * (r * a + a * r)).coeff
    )


def test_r_norm():
    r = ga.cga_object(np.random.rand(32))
    np.testing.assert_array_almost_equal(ga.r_norm(r).coeff, (r * (~r)).coeff)


def test_l_norm():
    r = ga.cga_object(np.random.rand(32))
    np.testing.assert_array_almost_equal(ga.l_norm(r).coeff, ((~r) * r).coeff)


def test_grade():
    a0 = np.zeros(32, dtype=complex)
    a1 = np.zeros(32, dtype=complex)
    a2 = np.zeros(32, dtype=complex)
    a3 = np.zeros(32, dtype=complex)
    a4 = np.zeros(32, dtype=complex)
    a5 = np.zeros(32, dtype=complex)
    a = ga.cga_object(np.random.rand(32))
    grade_indices = np.array(
        [
            [0],
            np.arange(1, 6),
            np.arange(6, 16),
            np.arange(16, 26),
            np.arange(26, 31),
            31,
        ],
        dtype=object,
    )

    a0[grade_indices[0]] = a.coeff[grade_indices[0]]
    a1[grade_indices[1]] = a.coeff[grade_indices[1]]
    a2[grade_indices[2]] = a.coeff[grade_indices[2]]
    a3[grade_indices[3]] = a.coeff[grade_indices[3]]
    a4[grade_indices[4]] = a.coeff[grade_indices[4]]
    a5[grade_indices[5]] = a.coeff[grade_indices[5]]
    b0 = ga.n_grade(a, 0).coeff
    b1 = ga.n_grade(a, 1).coeff
    b2 = ga.n_grade(a, 2).coeff
    b3 = ga.n_grade(a, 3).coeff
    b4 = ga.n_grade(a, 4).coeff
    b5 = ga.n_grade(a, 5).coeff
    np.testing.assert_array_almost_equal(a0, b0)
    np.testing.assert_array_almost_equal(a1, b1)
    np.testing.assert_array_almost_equal(a2, b2)
    np.testing.assert_array_almost_equal(a3, b3)
    np.testing.assert_array_almost_equal(a4, b4)
    np.testing.assert_array_almost_equal(a5, b5)
