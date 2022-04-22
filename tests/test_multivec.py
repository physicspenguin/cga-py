from cga_py import cga_object as cg
from cga_py import e_1, e_2, e_12
import numpy as np
from numpy.testing import assert_allclose


def test_mult_cga_1():
    assert e_1 * e_1 == 1


def test_mult_cga_2():
    assert e_1 * e_2 == e_12


def test_mult_cga_3():
    assert e_1 * e_12 == e_2


def test_mult_cga_4():
    assert e_12 * e_12 == -1


def test_vec_ident():
    size = 6
    end = 2000
    for _ in range(500):
        a = cg(np.random.rand(size) * end - (end / 2))
        b = cg(np.random.rand(size) * end - (end / 2))
        assert_allclose((a * b).coeff, (((a ^ b) + (a | b))).coeff)


test_vec_ident()
