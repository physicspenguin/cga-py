from cga_py import cga_object as cg
from cga_py import e_1, e_2, e_12
from cga_py.multivector import _mul, _inner, _wedge, _invert
import numpy as np
import numpy.testing as nt
import pytest as pt


def test_cg_gen():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    nt.assert_array_equal(cg(a).coeff, a)
    nt.assert_array_equal(cg(cg(a)).coeff, a)
    nt.assert_array_equal(cg([]).coeff, np.zeros(32))
    assert cg() == cg(0)


def test_cg_even_gen():
    a = cg(np.arange(16), even=True)
    nt.assert_array_equal(
        a.coeff,
        [
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            1.0 + 0.0j,
            2.0 + 0.0j,
            3.0 + 0.0j,
            4.0 + 0.0j,
            5.0 + 0.0j,
            6.0 + 0.0j,
            7.0 + 0.0j,
            8.0 + 0.0j,
            9.0 + 0.0j,
            10.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            11.0 + 0.0j,
            12.0 + 0.0j,
            13.0 + 0.0j,
            14.0 + 0.0j,
            15.0 + 0.0j,
            0.0 + 0.0j,
        ],
    )


def test_cg_error():
    with pt.raises(ValueError):
        assert cg("a")
    with pt.raises(ValueError):
        assert cg(["a", 2])
    with pt.raises(ValueError):
        assert cg(np.arange(33))


def test_mult_errors():
    with pt.raises(ValueError):
        assert cg(np.arange(32)) * "a"
    with pt.raises(ValueError):
        assert cg(np.arange(32)) * ["a"]
    with pt.raises(ValueError):
        assert cg(np.arange(32)) ^ "a"
    with pt.raises(ValueError):
        assert cg(np.arange(32)) ^ ["a"]
    with pt.raises(ValueError):
        assert cg(np.arange(32)) | "a"
    with pt.raises(ValueError):
        assert cg(np.arange(32)) | ["a"]


def test_mult_cga():
    assert e_1 * e_1 == 1
    assert e_1 * e_2 == e_12
    assert e_1 * e_12 == e_2
    assert e_12 * e_12 == -1


def test_add_cga():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    b = np.random.rand(32) * end - (end / 2)
    nt.assert_array_equal((cg(a) - cg(b)).coeff, a - b)


def test_sub_cga():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    b = np.random.rand(32) * end - (end / 2)
    nt.assert_array_equal((cg(a) - cg(b)).coeff, a - b)


def test_add_float():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    b = np.random.rand() * end - (end / 2)
    c = a.copy()
    c[0] = a[0] + b
    nt.assert_array_equal((cg(a) + b).coeff, c)


def test_sub_float():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    b = np.random.rand() * end - (end / 2)
    c = a.copy()
    d = -(a.copy())
    c[0] = a[0] - b
    d[0] = b - a[0]
    nt.assert_array_equal((cg(a) - b).coeff, c)
    nt.assert_array_equal((b - cg(a)).coeff, d)


def test_div_float():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    b = np.random.rand() * end - (end / 2)
    nt.assert_allclose((cg(a) / b).coeff, a / b)
    with pt.raises(TypeError):
        assert cg(a) / cg(b)


def test_floor_div_float():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    b = np.random.rand() * end - (end / 2)
    nt.assert_array_equal((cg(a) // b).coeff, a // b)
    with pt.raises(TypeError):
        assert cg(a) // cg(b)


def test_wedge():
    a = np.array(
        [
            844.94424921,
            200.2260122,
            565.29703616,
            517.1192215,
            548.71211879,
            -982.0077644,
            434.89547631,
            930.81244647,
            721.32643468,
            -683.79338806,
            256.38840576,
            -355.69394934,
            681.53327362,
            -22.67814807,
            -107.08448605,
            -40.46962442,
            -996.95261258,
            830.73946602,
            777.66477061,
            716.48772892,
            959.13937444,
            -904.77431682,
            68.89817051,
            745.33768046,
            -109.06817517,
            -966.88535839,
            -488.53225325,
            602.89326163,
            375.97793653,
            -250.05151507,
            828.61258309,
            737.38156624,
        ]
    )
    b = np.array(
        [
            725.11011632,
            -975.08265755,
            144.70651935,
            38.38948582,
            -471.52565452,
            -635.02557765,
            830.61336759,
            224.20132730,
            707.70612438,
            899.80033113,
            68.54020623,
            408.10948996,
            960.27688338,
            88.25762321,
            726.74012486,
            55.61095813,
            -652.38752758,
            545.21429166,
            -966.77157658,
            -899.58613452,
            559.06038443,
            -587.10321068,
            -298.20056367,
            -758.32329002,
            446.03370508,
            -616.11082366,
            342.97958509,
            337.41571782,
            416.78904346,
            498.01600767,
            139.74679988,
            755.44123826,
        ]
    )
    c = np.array(
        [
            -196559.9066,
            -2.621340173 * 10**6,
            175840.6706,
            -85291.66173,
            -50133.0546,
            -1.219713828 * 10**6,
            130866.3544,
            312025.2015,
            801393.2829,
            -743805.7028,
            -408862.8069,
            -29429.6144,
            582383.6343,
            -995916.9588,
            272116.3361,
            -789343.7496,
            -1.039655143 * 10**6,
            1.567274899 * 10**6,
            -795294.1106,
            -2.439895344 * 10**6,
            612981.9762,
            -3.017614578 * 10**6,
            -293640.3294,
            -1.087887902 * 10**6,
            -8847.0000,
            -1.658335133 * 10**6,
            880405.0841,
            -4.152271715 * 10**6,
            -203657.8889,
            -1.535109243 * 10**6,
            -851255.0594,
            31507.0908,
        ]
    )
    nt.assert_allclose((cg(a) ^ cg(b)).coeff, c)
    nt.assert_allclose((cg(a) ^ b).coeff, c)
    nt.assert_allclose(_wedge(a, b), c)


def test_inner():
    a = np.array(
        [
            844.94424921,
            200.2260122,
            565.29703616,
            517.1192215,
            548.71211879,
            -982.0077644,
            434.89547631,
            930.81244647,
            721.32643468,
            -683.79338806,
            256.38840576,
            -355.69394934,
            681.53327362,
            -22.67814807,
            -107.08448605,
            -40.46962442,
            -996.95261258,
            830.73946602,
            777.66477061,
            716.48772892,
            959.13937444,
            -904.77431682,
            68.89817051,
            745.33768046,
            -109.06817517,
            -966.88535839,
            -488.53225325,
            602.89326163,
            375.97793653,
            -250.05151507,
            828.61258309,
            737.38156624,
        ]
    )
    b = np.array(
        [
            725.11011632,
            -975.08265755,
            144.70651935,
            38.38948582,
            -471.52565452,
            -635.02557765,
            830.61336759,
            224.20132730,
            707.70612438,
            899.80033113,
            68.54020623,
            408.10948996,
            960.27688338,
            88.25762321,
            726.74012486,
            55.61095813,
            -652.38752758,
            545.21429166,
            -966.77157658,
            -899.58613452,
            559.06038443,
            -587.10321068,
            -298.20056367,
            -758.32329002,
            446.03370508,
            -616.11082366,
            342.97958509,
            337.41571782,
            416.78904346,
            498.01600767,
            139.74679988,
            755.44123826,
        ]
    )
    c = np.array(
        [
            -2.530411112 * 10**6,
            1.599954023 * 10**6,
            1.822826693 * 10**6,
            -2.696604087 * 10**6,
            4.078937378 * 10**6,
            1.987185971 * 10**6,
            -1.187436685 * 10**6,
            -3.145964557 * 10**5,
            6.660333064 * 10**5,
            1.222146679 * 10**6,
            -8.956764629 * 10**5,
            1.539030775 * 10**6,
            8.028722657 * 10**5,
            70565.28404,
            2.593055265 * 10**5,
            1.393631531 * 10**5,
            -1.671812865 * 10**5,
            1.186760692 * 10**6,
            7.316558495 * 10**5,
            1.035028005 * 10**5,
            -9.843136698 * 10**5,
            -8.122890646 * 10**5,
            -4.099242511 * 10**5,
            3.647273854 * 10**5,
            1.631533479 * 10**6,
            4.475617144 * 10**5,
            -66825.4369,
            -1.210105317 * 10**6,
            1.603694721 * 10**6,
            -2.872501782 * 10**5,
            5.486739247 * 10**5,
            0,
        ]
    )
    nt.assert_allclose((cg(a) | cg(b)).coeff, c)
    nt.assert_allclose((cg(a) | b).coeff, c)
    nt.assert_allclose(_inner(a, b), c)


def test_mult():
    a = np.array(
        [
            844.94424921,
            200.2260122,
            565.29703616,
            517.1192215,
            548.71211879,
            -982.0077644,
            434.89547631,
            930.81244647,
            721.32643468,
            -683.79338806,
            256.38840576,
            -355.69394934,
            681.53327362,
            -22.67814807,
            -107.08448605,
            -40.46962442,
            -996.95261258,
            830.73946602,
            777.66477061,
            716.48772892,
            959.13937444,
            -904.77431682,
            68.89817051,
            745.33768046,
            -109.06817517,
            -966.88535839,
            -488.53225325,
            602.89326163,
            375.97793653,
            -250.05151507,
            828.61258309,
            737.38156624,
        ]
    )
    b = np.array(
        [
            725.11011632,
            -975.08265755,
            144.70651935,
            38.38948582,
            -471.52565452,
            -635.02557765,
            830.61336759,
            224.20132730,
            707.70612438,
            899.80033113,
            68.54020623,
            408.10948996,
            960.27688338,
            88.25762321,
            726.74012486,
            55.61095813,
            -652.38752758,
            545.21429166,
            -966.77157658,
            -899.58613452,
            559.06038443,
            -587.10321068,
            -298.20056367,
            -758.32329002,
            446.03370508,
            -616.11082366,
            342.97958509,
            337.41571782,
            416.78904346,
            498.01600767,
            139.74679988,
            755.44123826,
        ]
    )
    c = np.array(
        [
            -3.811101289 * 10**6,
            -2.095686347 * 10**6,
            1.690338753 * 10**6,
            -466728.5936,
            4.028804322 * 10**6,
            767472.1417,
            2.298739138 * 10**6,
            1.191732144 * 10**6,
            2.424952035 * 10**6,
            -1.285441671 * 10**6,
            -27990.38776,
            4.458706304 * 10**6,
            683967.3284,
            -2.825153081 * 10**6,
            -2.705930939 * 10**6,
            -1.734110866 * 10**6,
            -2.538560167 * 10**6,
            4.500299091 * 10**6,
            -2.375924537 * 10**6,
            -668380.3831,
            353681.3339,
            -4.904203838 * 10**6,
            -610622.9800,
            -3.984941706 * 10**6,
            1.314357869 * 10**6,
            1.104393736 * 10**6,
            1.574466290 * 10**6,
            -3.333694615 * 10**6,
            4.480594744 * 10**6,
            -2.319311963 * 10**6,
            460431.7540,
            31507.0906,
        ]
    )
    nt.assert_allclose((cg(a) * cg(b)).coeff, c)
    nt.assert_allclose((cg(a) * b).coeff, c)
    nt.assert_allclose(_mul(a, b), c)


def test_vec_ident():
    size = 6
    end = 2000
    for _ in range(100):
        a = cg(np.random.rand(size) * end - (end / 2))
        b = cg(np.random.rand(size) * end - (end / 2))
        nt.assert_allclose((a * b).coeff, (((a ^ b) + (a | b))).coeff)


def test_pos():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    nt.assert_array_equal((+cg(a)).coeff, a)


def test_neg():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    nt.assert_array_equal((-cg(a)).coeff, -a)


def test_inv():
    a = np.array(
        [
            844.94424921,
            200.2260122,
            565.29703616,
            517.1192215,
            548.71211879,
            -982.0077644,
            434.89547631,
            930.81244647,
            721.32643468,
            -683.79338806,
            256.38840576,
            -355.69394934,
            681.53327362,
            -22.67814807,
            -107.08448605,
            -40.46962442,
            -996.95261258,
            830.73946602,
            777.66477061,
            716.48772892,
            959.13937444,
            -904.77431682,
            68.89817051,
            745.33768046,
            -109.06817517,
            -966.88535839,
            -488.53225325,
            602.89326163,
            375.97793653,
            -250.05151507,
            828.61258309,
            737.38156624,
        ]
    )
    b = np.array(
        [
            925.8834980,
            2009.774646,
            783.4333866,
            2450.889938,
            548.71211879,
            -982.0077644,
            317.0603967,
            -1430.915477,
            -721.32643468,
            683.79338806,
            1400.836760,
            355.69394934,
            -681.53327362,
            22.67814807,
            107.08448605,
            40.46962442,
            2471.715745,
            -830.73946602,
            -777.66477061,
            -716.48772892,
            -959.13937444,
            904.77431682,
            -68.89817051,
            -745.33768046,
            109.06817517,
            966.88535839,
            -488.53225325,
            602.89326163,
            375.97793653,
            -250.05151507,
            828.61258309,
            737.38156624,
        ]
    )
    nt.assert_allclose((~cg(a)).coeff, b)
    nt.assert_allclose(_invert(a), b)


def test_eq():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    assert (cg(a) == cg(a)) == True
    assert (cg(a) == cg(a - 1)) == False
    assert (cg(a) == 3000) == False
    assert (cg(a) == "a") == False


def test_neq():
    end = 2000
    a = np.random.rand(32) * end - (end / 2)
    assert (cg(a) != cg(a)) == False
    assert (cg(a) != cg(a - 1)) == True
    assert (cg(a) != 3000) == True
    assert (cg(a) != "a") == True


def test_str():
    a = cg(np.arange(32))
    assert (
        str(a)
        == "np.complex128(1+0j)e_1 + np.complex128(2+0j)e_2 + np.complex128(3+0j)e_3 + np.complex128(4+0j)e_i + np.complex128(5+0j)e_o + np.complex128(6+0j)e_12 + np.complex128(7+0j)e_13 + np.complex128(8+0j)e_1i + np.complex128(9+0j)e_1o + np.complex128(10+0j)e_23 + np.complex128(11+0j)e_2i + np.complex128(12+0j)e_2o + np.complex128(13+0j)e_3i + np.complex128(14+0j)e_3o + np.complex128(15+0j)e_io + np.complex128(16+0j)e_123 + np.complex128(17+0j)e_12i + np.complex128(18+0j)e_12o + np.complex128(19+0j)e_13i + np.complex128(20+0j)e_13o + np.complex128(21+0j)e_1io + np.complex128(22+0j)e_23i + np.complex128(23+0j)e_23o + np.complex128(24+0j)e_2io + np.complex128(25+0j)e_3io + np.complex128(26+0j)e_123i + np.complex128(27+0j)e_123o + np.complex128(28+0j)e_12io + np.complex128(29+0j)e_13io + np.complex128(30+0j)e_23io + np.complex128(31+0j)e_123io"
    )
    assert str(cg()) == "0"


def test_repr():
    a = cg(np.arange(32))
    assert (
        repr(a)
        == "np.complex128(1+0j)*e_1 + np.complex128(2+0j)*e_2 + np.complex128(3+0j)*e_3 + np.complex128(4+0j)*e_i + np.complex128(5+0j)*e_o + np.complex128(6+0j)*e_12 + np.complex128(7+0j)*e_13 + np.complex128(8+0j)*e_1i + np.complex128(9+0j)*e_1o + np.complex128(10+0j)*e_23 + np.complex128(11+0j)*e_2i + np.complex128(12+0j)*e_2o + np.complex128(13+0j)*e_3i + np.complex128(14+0j)*e_3o + np.complex128(15+0j)*e_io + np.complex128(16+0j)*e_123 + np.complex128(17+0j)*e_12i + np.complex128(18+0j)*e_12o + np.complex128(19+0j)*e_13i + np.complex128(20+0j)*e_13o + np.complex128(21+0j)*e_1io + np.complex128(22+0j)*e_23i + np.complex128(23+0j)*e_23o + np.complex128(24+0j)*e_2io + np.complex128(25+0j)*e_3io + np.complex128(26+0j)*e_123i + np.complex128(27+0j)*e_123o + np.complex128(28+0j)*e_12io + np.complex128(29+0j)*e_13io + np.complex128(30+0j)*e_23io + np.complex128(31+0j)*e_123io"
    )
    assert repr(cg()) == "0"


def test_make_even():
    a = cg(np.arange(32))
    assert all(
        a.get_even()
        == [
            0.0 + 0.0j,
            6.0 + 0.0j,
            7.0 + 0.0j,
            8.0 + 0.0j,
            9.0 + 0.0j,
            10.0 + 0.0j,
            11.0 + 0.0j,
            12.0 + 0.0j,
            13.0 + 0.0j,
            14.0 + 0.0j,
            15.0 + 0.0j,
            26.0 + 0.0j,
            27.0 + 0.0j,
            28.0 + 0.0j,
            29.0 + 0.0j,
            30.0 + 0.0j,
        ]
    )


def test_get_even():
    a = cg(np.arange(32))
    assert a.make_even() == cg(
        [
            0.0 + 0.0j,
            6.0 + 0.0j,
            7.0 + 0.0j,
            8.0 + 0.0j,
            9.0 + 0.0j,
            10.0 + 0.0j,
            11.0 + 0.0j,
            12.0 + 0.0j,
            13.0 + 0.0j,
            14.0 + 0.0j,
            15.0 + 0.0j,
            26.0 + 0.0j,
            27.0 + 0.0j,
            28.0 + 0.0j,
            29.0 + 0.0j,
            30.0 + 0.0j,
        ],
        even=True,
    )


def test_complex_floordiv():
    with nt.assert_raises(TypeError):
        cg(1j * np.random.rand(32)) // (4)
