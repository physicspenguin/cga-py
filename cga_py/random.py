import numpy.random as rand
from .multivector import *
from .base_objects import *
from .geom_generators import *
from .conditions import *


def rand_rational(maximum=10):
    """Generate random rational number up to n

    Kwargs:
        maximum (int): Maximum of random number generated

    Returns: (float) Random rational number in Range maximum

    Parameters
    ----------
    maximum :
         (Default value = 10)

    Returns
    -------

    """
    return (-1) ** (rand.randint(2)) * (
        rand.randint(maximum) / (rand.randint(maximum) + 1)
    )


def rand_point(maximum=10, conformal=True):
    """TODO: Generate random Point with rational coefficients

    Kwargs:
        maximum (int): Maximum of random number generated
        conformal (bool): Is conformal representation to be used

    Returns: cga_object

    Parameters
    ----------
    maximum :
         (Default value = 10)
    conformal :
         (Default value = True)

    Returns
    -------

    """
    return point(
        [rand_rational(maximum), rand_rational(maximum), rand_rational(maximum)],
        conformal,
    )


def rand_sphere(maximum=10):
    """Generate random Sphere with rational parameters

    Kwargs:
        maximum (int): Maximum of rational parameters

    Returns: cga_object

    Parameters
    ----------
    maximum :
         (Default value = 10)

    Returns
    -------

    """
    return sphere(
        [rand_rational(maximum), rand_rational(maximum), rand_rational(maximum)],
        rand_rational(maximum),
    )


def rand_plane(maximum=10):
    """Generate random Plane with rational parameters

    Kwargs:
        maximum (int): Maximum of rational parameters

    Returns: cga_object

    Parameters
    ----------
    maximum :
         (Default value = 10)

    Returns
    -------

    """
    return plane(
        [rand_rational(maximum), rand_rational(maximum), rand_rational(maximum)],
        rand_rational(maximum),
    )


def rand_rotor(maximum=10, tol=0):
    """Generates random rotor with rational coefficients

    Kwargs:
        maximum (int): maximum for rational coefficients
        tol (float): tolerance for numerical error

    Returns: cga-object

    Parameters
    ----------
    maximum :
         (Default value = 10)
    tol :
         (Default value = 0)

    Returns
    -------

    """
    a = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
        ]
    )

    if a[15] == 0:
        return rand_rotor(maximum)
    a[0] = (a[5] * a[10] - a[6] * a[9] + a[7] * a[8]) / a[15]
    a[1] = (a[5] * a[13] - a[6] * a[12] + a[7] * a[11]) / a[15]
    a[2] = (a[5] * a[14] - a[8] * a[12] + a[9] * a[11]) / a[15]
    a[3] = (a[6] * a[14] - a[8] * a[13] + a[10] * a[11]) / a[15]
    a[4] = (a[7] * a[14] - a[9] * a[13] + a[10] * a[12]) / a[15]
    out = cga_object(a, True)
    if not (np.linalg.norm(study_var(out)) <= tol):
        return rand_rotor(maximum)

    return out


def rand_rot_poly(maximum=10, tol=0):
    """Generates random rotor for rotor polynomials with rational coefficients.

    Kwargs:
        maximum (int): maximum for rational coefficients
        tol (float): tolerance for numerical error

    Returns: cga-object

    Parameters
    ----------
    maximum :
         (Default value = 10)
    tol :
         (Default value = 0)

    Returns
    -------

    """
    a = np.array(
        [
            rand_rational(maximum),
            0,
            0,
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            0,
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            rand_rational(maximum),
            0,
            0,
            0,
            0,
            0,
        ]
    )

    if a[10] == 0:
        return rand_rot_poly(maximum)
    a[1] = (a[3] * a[7] - a[4] * a[6]) / a[10]
    a[2] = (a[3] * a[9] - a[4] * a[8]) / a[10]
    a[5] = (a[6] * a[9] - a[7] * a[8]) / a[10]
    out = cga_object(a, True)
    if np.linalg.norm(study_var(out)) > tol:
        return rand_rot_poly(maximum)
    return out


def rand_zero(maximum=10, tol=0):
    """Generates random rotor for zero displacement

    Kwargs:
        maximum (int): maximum for rational coefficients
        tol (float): tolerance for numerical error

    Returns: cga-object

    Parameters
    ----------
    maximum :
         (Default value = 10)
    tol :
         (Default value = 0)

    Returns
    -------

    """

    def gen():
        """ """
        b = np.array(
            [
                rand_rational(maximum),
                rand_rational(maximum),
                rand_rational(maximum),
                rand_rational(maximum),
                rand_rational(maximum),
                rand_rational(maximum),
                rand_rational(maximum),
                rand_rational(maximum),
                rand_rational(maximum),
                rand_rational(maximum),
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )

        # Protect against 0 division
        while (
            (2 * b[-16] ** 2 * (b[-11] ** 2 + b[-14] ** 2 + b[-15] ** 2 + b[-16] ** 2))
            == 0
            or (2 * b[-16] * (b[-11] ** 2 + b[-14] ** 2 + b[-15] ** 2 + b[-16] ** 2))
            == 0
            or b[-16] == 0
        ):
            b = np.array(
                [
                    rand_rational(maximum),
                    rand_rational(maximum),
                    rand_rational(maximum),
                    rand_rational(maximum),
                    rand_rational(maximum),
                    rand_rational(maximum),
                    rand_rational(maximum),
                    rand_rational(maximum),
                    rand_rational(maximum),
                    rand_rational(maximum),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
        b[-1] = -(
            2 * b[-7] * b[-8] * b[-11] * b[-15] ** 2
            + 2 * b[-7] * b[-8] * b[-11] * b[-16] ** 2
            - 2 * b[-7] * b[-10] * b[-11] * b[-14] * b[-15]
            + 2 * b[-7] * b[-10] * b[-14] ** 2 * b[-16]
            + 2 * b[-7] * b[-10] * b[-15] ** 2 * b[-16]
            + 2 * b[-7] * b[-10] * b[-16] ** 3
            + 2 * b[-7] * b[-11] ** 2 * b[-13] * b[-15]
            - 2 * b[-7] * b[-11] * b[-13] * b[-14] * b[-16]
            - 2 * b[-8] * b[-9] * b[-11] * b[-14] * b[-15]
            - 2 * b[-8] * b[-9] * b[-14] ** 2 * b[-16]
            - 2 * b[-8] * b[-9] * b[-15] ** 2 * b[-16]
            - 2 * b[-8] * b[-9] * b[-16] ** 3
            + 2 * b[-8] * b[-11] ** 2 * b[-12] * b[-15]
            + 2 * b[-8] * b[-11] * b[-12] * b[-14] * b[-16]
            + 2 * b[-9] * b[-10] * b[-11] * b[-14] ** 2
            + 2 * b[-9] * b[-10] * b[-11] * b[-16] ** 2
            - 2 * b[-9] * b[-11] ** 2 * b[-13] * b[-14]
            - 2 * b[-9] * b[-11] * b[-13] * b[-15] * b[-16]
            - 2 * b[-10] * b[-11] ** 2 * b[-12] * b[-14]
            + 2 * b[-10] * b[-11] * b[-12] * b[-15] * b[-16]
            + 2 * b[-11] ** 3 * b[-12] * b[-13]
            - b[-11] ** 3 * b[-16] ** 2
            + 2 * b[-11] * b[-12] * b[-13] * b[-16] ** 2
            - b[-11] * b[-14] ** 2 * b[-16] ** 2
            - b[-11] * b[-15] ** 2 * b[-16] ** 2
            - b[-11] * b[-16] ** 4
        ) / (2 * b[-16] ** 2 * (b[-11] ** 2 + b[-14] ** 2 + b[-15] ** 2 + b[-16] ** 2))
        b[-2] = -(
            2 * b[-7] * b[-8] * b[-14] * b[-15] ** 2
            + 2 * b[-7] * b[-8] * b[-14] * b[-16] ** 2
            - 2 * b[-7] * b[-10] * b[-11] * b[-14] * b[-16]
            - 2 * b[-7] * b[-10] * b[-14] ** 2 * b[-15]
            + 2 * b[-7] * b[-11] ** 2 * b[-13] * b[-16]
            + 2 * b[-7] * b[-11] * b[-13] * b[-14] * b[-15]
            + 2 * b[-7] * b[-13] * b[-15] ** 2 * b[-16]
            + 2 * b[-7] * b[-13] * b[-16] ** 3
            + 2 * b[-8] * b[-9] * b[-11] * b[-14] * b[-16]
            - 2 * b[-8] * b[-9] * b[-14] ** 2 * b[-15]
            - 2 * b[-8] * b[-11] ** 2 * b[-12] * b[-16]
            + 2 * b[-8] * b[-11] * b[-12] * b[-14] * b[-15]
            - 2 * b[-8] * b[-12] * b[-15] ** 2 * b[-16]
            - 2 * b[-8] * b[-12] * b[-16] ** 3
            + 2 * b[-9] * b[-10] * b[-14] ** 3
            + 2 * b[-9] * b[-10] * b[-14] * b[-16] ** 2
            - 2 * b[-9] * b[-11] * b[-13] * b[-14] ** 2
            - 2 * b[-9] * b[-13] * b[-14] * b[-15] * b[-16]
            - 2 * b[-10] * b[-11] * b[-12] * b[-14] ** 2
            + 2 * b[-10] * b[-12] * b[-14] * b[-15] * b[-16]
            + 2 * b[-11] ** 2 * b[-12] * b[-13] * b[-14]
            - b[-11] ** 2 * b[-14] * b[-16] ** 2
            + 2 * b[-12] * b[-13] * b[-14] * b[-16] ** 2
            - b[-14] ** 3 * b[-16] ** 2
            - b[-14] * b[-15] ** 2 * b[-16] ** 2
            - b[-14] * b[-16] ** 4
        ) / (2 * b[-16] ** 2 * (b[-11] ** 2 + b[-14] ** 2 + b[-15] ** 2 + b[-16] ** 2))
        b[-3] = -(
            2 * b[-7] * b[-8] * b[-15] ** 3
            + 2 * b[-7] * b[-8] * b[-15] * b[-16] ** 2
            - 2 * b[-7] * b[-10] * b[-11] * b[-15] * b[-16]
            - 2 * b[-7] * b[-10] * b[-14] * b[-15] ** 2
            + 2 * b[-7] * b[-11] * b[-13] * b[-15] ** 2
            - 2 * b[-7] * b[-13] * b[-14] * b[-15] * b[-16]
            + 2 * b[-8] * b[-9] * b[-11] * b[-15] * b[-16]
            - 2 * b[-8] * b[-9] * b[-14] * b[-15] ** 2
            + 2 * b[-8] * b[-11] * b[-12] * b[-15] ** 2
            + 2 * b[-8] * b[-12] * b[-14] * b[-15] * b[-16]
            + 2 * b[-9] * b[-10] * b[-14] ** 2 * b[-15]
            + 2 * b[-9] * b[-10] * b[-15] * b[-16] ** 2
            + 2 * b[-9] * b[-11] ** 2 * b[-13] * b[-16]
            - 2 * b[-9] * b[-11] * b[-13] * b[-14] * b[-15]
            + 2 * b[-9] * b[-13] * b[-14] ** 2 * b[-16]
            + 2 * b[-9] * b[-13] * b[-16] ** 3
            - 2 * b[-10] * b[-11] ** 2 * b[-12] * b[-16]
            - 2 * b[-10] * b[-11] * b[-12] * b[-14] * b[-15]
            - 2 * b[-10] * b[-12] * b[-14] ** 2 * b[-16]
            - 2 * b[-10] * b[-12] * b[-16] ** 3
            + 2 * b[-11] ** 2 * b[-12] * b[-13] * b[-15]
            - b[-11] ** 2 * b[-15] * b[-16] ** 2
            + 2 * b[-12] * b[-13] * b[-15] * b[-16] ** 2
            - b[-14] ** 2 * b[-15] * b[-16] ** 2
            - b[-15] ** 3 * b[-16] ** 2
            - b[-15] * b[-16] ** 4
        ) / (2 * b[-16] ** 2 * (b[-11] ** 2 + b[-14] ** 2 + b[-15] ** 2 + b[-16] ** 2))
        b[-4] = (b[-7] * b[-15] - b[-9] * b[-14] + b[-11] * b[-12]) / b[-16]
        b[-5] = (b[-8] * b[-15] - b[-10] * b[-14] + b[-11] * b[-13]) / b[-16]
        b[-6] = -(
            2 * b[-7] * b[-8] * b[-15] ** 2
            + 2 * b[-7] * b[-8] * b[-16] ** 2
            - 2 * b[-7] * b[-10] * b[-11] * b[-16]
            - 2 * b[-7] * b[-10] * b[-14] * b[-15]
            + 2 * b[-7] * b[-11] * b[-13] * b[-15]
            - 2 * b[-7] * b[-13] * b[-14] * b[-16]
            + 2 * b[-8] * b[-9] * b[-11] * b[-16]
            - 2 * b[-8] * b[-9] * b[-14] * b[-15]
            + 2 * b[-8] * b[-11] * b[-12] * b[-15]
            + 2 * b[-8] * b[-12] * b[-14] * b[-16]
            + 2 * b[-9] * b[-10] * b[-14] ** 2
            + 2 * b[-9] * b[-10] * b[-16] ** 2
            - 2 * b[-9] * b[-11] * b[-13] * b[-14]
            - 2 * b[-9] * b[-13] * b[-15] * b[-16]
            - 2 * b[-10] * b[-11] * b[-12] * b[-14]
            + 2 * b[-10] * b[-12] * b[-15] * b[-16]
            + 2 * b[-11] ** 2 * b[-12] * b[-13]
            - b[-11] ** 2 * b[-16] ** 2
            + 2 * b[-12] * b[-13] * b[-16] ** 2
            - b[-14] ** 2 * b[-16] ** 2
            - b[-15] ** 2 * b[-16] ** 2
            - b[-16] ** 4
        ) / (2 * b[-16] * (b[-11] ** 2 + b[-14] ** 2 + b[-15] ** 2 + b[-16] ** 2))
        return b

    while True:
        b = gen()
        out = cga_object(b, True)
        # Check if Null displacement condition and Study condition is fullfilled
        if max(abs(study_var(out))) <= tol and abs(null_quadric(out)) <= tol:
            return out
