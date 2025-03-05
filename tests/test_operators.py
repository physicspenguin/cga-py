import cga_py as ga
import numpy as np
from cga_py.permutations_pm import (
    e_12,
    e_13,
    e_1p,
    e_1m,
    e_23,
    e_2p,
    e_2m,
    e_3p,
    e_3m,
    e_pm,
    e_123,
    e_12p,
    e_12m,
    e_13p,
    e_13m,
    e_1pm,
    e_23p,
    e_23m,
    e_2pm,
    e_3pm,
    e_123p,
    e_123m,
    e_12pm,
    e_13pm,
    e_23pm,
    e_123pm,
)
from cga_py.base_objects import e_1, e_2, e_3, e_p, e_m


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
    a = np.random.rand(32)
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

    def pm_vec_to_obj(vec):
        pm_base = [
            1,
            e_1,
            e_2,
            e_3,
            e_p,
            e_m,
            e_12,
            e_13,
            e_1p,
            e_1m,
            e_23,
            e_2p,
            e_2m,
            e_3p,
            e_3m,
            e_pm,
            e_123,
            e_12p,
            e_12m,
            e_13p,
            e_13m,
            e_1pm,
            e_23p,
            e_23m,
            e_2pm,
            e_3pm,
            e_123p,
            e_123m,
            e_12pm,
            e_13pm,
            e_23pm,
            e_123pm,
        ]
        out = 0
        for i, v in enumerate(vec):
            out += v * pm_base[i]
        return out

    def io_to_pm_vec(obj):
        h = ga.cga_object(obj).coeff
        return [
            (-h[15] + h[0]),
            (h[1] - h[21]),
            (h[2] - h[24]),
            (h[3] - h[25]),
            (-1 / 2 * h[5] + h[4]),
            (h[4] + 1 / 2 * h[5]),
            (h[6] - h[28]),
            (h[7] - h[29]),
            (-1 / 2 * h[9] + h[8]),
            (h[8] + 1 / 2 * h[9]),
            (h[10] - h[30]),
            (-1 / 2 * h[12] + h[11]),
            (h[11] + 1 / 2 * h[12]),
            (-1 / 2 * h[14] + h[13]),
            (h[13] + 1 / 2 * h[14]),
            (h[15]),
            (h[16] - h[31]),
            (-1 / 2 * h[18] + h[17]),
            (h[17] + 1 / 2 * h[18]),
            (-1 / 2 * h[20] + h[19]),
            (h[19] + 1 / 2 * h[20]),
            (h[21]),
            (-1 / 2 * h[23] + h[22]),
            (h[22] + 1 / 2 * h[23]),
            (h[24]),
            (h[25]),
            (-1 / 2 * h[27] + h[26]),
            (h[26] + 1 / 2 * h[27]),
            (h[28]),
            (h[29]),
            (h[30]),
            (h[31]),
        ]

    np.testing.assert_array_almost_equal(
        ga.cga_object(a).coeff,
        ga.cga_object(pm_vec_to_obj(io_to_pm_vec(ga.cga_object(a)))).coeff,
    )

    a0[grade_indices[0]] = a[grade_indices[0]]
    a1[grade_indices[1]] = a[grade_indices[1]]
    a2[grade_indices[2]] = a[grade_indices[2]]
    a3[grade_indices[3]] = a[grade_indices[3]]
    a4[grade_indices[4]] = a[grade_indices[4]]
    a5[grade_indices[5]] = a[grade_indices[5]]
    c = pm_vec_to_obj(a)
    b0 = io_to_pm_vec(ga.n_grade(c, 0))
    b1 = io_to_pm_vec(ga.n_grade(c, 1))
    b2 = io_to_pm_vec(ga.n_grade(c, 2))
    b3 = io_to_pm_vec(ga.n_grade(c, 3))
    b4 = io_to_pm_vec(ga.n_grade(c, 4))
    b5 = io_to_pm_vec(ga.n_grade(c, 5))
    np.testing.assert_array_almost_equal(a0, b0)
    np.testing.assert_array_almost_equal(a1, b1)
    np.testing.assert_array_almost_equal(a2, b2)
    np.testing.assert_array_almost_equal(a3, b3)
    np.testing.assert_array_almost_equal(a4, b4)
    np.testing.assert_array_almost_equal(a5, b5)


# def test_grade():
#     a0 = np.zeros(32, dtype=complex)
#     a1 = np.zeros(32, dtype=complex)
#     a2 = np.zeros(32, dtype=complex)
#     a3 = np.zeros(32, dtype=complex)
#     a4 = np.zeros(32, dtype=complex)
#     a5 = np.zeros(32, dtype=complex)
#     a = ga.cga_object(np.random.rand(32))
#     grade_indices = np.array(
#         [
#             [0],
#             np.arange(1, 6),
#             np.arange(6, 16),
#             np.arange(16, 26),
#             np.arange(26, 31),
#             31,
#         ],
#         dtype=object,
#     )
#
#     a0[grade_indices[0]] = a.coeff[grade_indices[0]]
#     a1[grade_indices[1]] = a.coeff[grade_indices[1]]
#     a2[grade_indices[2]] = a.coeff[grade_indices[2]]
#     a3[grade_indices[3]] = a.coeff[grade_indices[3]]
#     a4[grade_indices[4]] = a.coeff[grade_indices[4]]
#     a5[grade_indices[5]] = a.coeff[grade_indices[5]]
#     b0 = ga.n_grade(a, 0).coeff
#     b1 = ga.n_grade(a, 1).coeff
#     b2 = ga.n_grade(a, 2).coeff
#     b3 = ga.n_grade(a, 3).coeff
#     b4 = ga.n_grade(a, 4).coeff
#     b5 = ga.n_grade(a, 5).coeff
#     np.testing.assert_array_almost_equal(a0, b0)
#     np.testing.assert_array_almost_equal(a1, b1)
#     np.testing.assert_array_almost_equal(a2, b2)
#     np.testing.assert_array_almost_equal(a3, b3)
#     np.testing.assert_array_almost_equal(a4, b4)
#     np.testing.assert_array_almost_equal(a5, b5)
