import numpy.random as rand
import numpy as np
from .multivector import cga_object
from .geom_generators import point, sphere, plane
from .conditions import study_var, null_quadric


def rand_rational(maximum=10):
    """Generate random rational number up to n

    Parameters
    ----------
    maximum : int (optional)
        Maximum of random number generated
         (Default value = 10)

    Returns
    -------
    (float)
        Random rational number in Range maximum

    """
    return (-1) ** (rand.randint(2)) * (
        rand.randint(maximum) / (rand.randint(maximum) + 1)
    )


def rand_point(maximum=10):
    """TODO: Generate random point with rational coefficients

    Parameters
    ----------
    maximum : int (optional)
        Maximum of random number generated
         (Default value = 10)

    Returns
    -------
    (cga_object)
        Random point

    """
    return point(
        [rand_rational(maximum), rand_rational(maximum), rand_rational(maximum)]
    )


def rand_sphere(maximum=10):
    """Generate random sphere with rational parameters

    Parameters
    ----------
    maximum : int (optional)
        Maximum of rational parameters
         (Default value = 10)

    Returns
    -------
    (cga_object)
        Random sphere

    """
    return sphere(
        [rand_rational(maximum), rand_rational(maximum), rand_rational(maximum)],
        rand_rational(maximum),
    )


def rand_plane(maximum=10):
    """Generate random Plane with rational parameters

    Parameters
    ----------
    maximum : int (optional)
        Maximum of rational parameters
         (Default value = 10)

    Returns
    -------
    (cga_object)
        Random plane

    """
    return plane(
        [rand_rational(maximum), rand_rational(maximum), rand_rational(maximum)],
        rand_rational(maximum),
    )


def rand_rotor(maximum=10, tol=0):
    """Generates random rotor with rational coefficients

    Parameters
    ----------
    maximum : int (optional)
        Maximum for rational coefficients
         (Default value = 10)
    tol : float (optional)
        Tolerance for numerical error
         (Default value = 0)

    Returns
    -------
    (cga_object)
        Random rotor

    """
    while True:
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
            continue
        a[0] = (a[5] * a[10] - a[6] * a[9] + a[7] * a[8]) / a[15]
        a[1] = (a[5] * a[13] - a[6] * a[12] + a[7] * a[11]) / a[15]
        a[2] = (a[5] * a[14] - a[8] * a[12] + a[9] * a[11]) / a[15]
        a[3] = (a[6] * a[14] - a[8] * a[13] + a[10] * a[11]) / a[15]
        a[4] = (a[7] * a[14] - a[9] * a[13] + a[10] * a[12]) / a[15]
        out = cga_object(a, True)
        # This could be replaced by numpy.testing.is_close in the future
        if not (np.linalg.norm(study_var(out)) <= tol):
            continue

        return out


def rand_rot_poly(maximum=10, tol=0):
    """Generates random rotor for rotor polynomials with rational coefficients.

    Parameters
    ----------
    maximum : int (optional)
        Maximum for rational coefficients
         (Default value = 10)
    tol : float (optional)
        Tolerance for numerical error
         (Default value = 0)

    Returns
    -------
    (cga_object)
        random rotor polynomial

    """
    while True:
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

        if (
            a[10] == 0
        ):  # pragma: no cover (because it is impossible to test this consistently)
            continue
        a[1] = (a[3] * a[7] - a[4] * a[6]) / a[10]
        a[2] = (a[3] * a[9] - a[4] * a[8]) / a[10]
        a[5] = (a[6] * a[9] - a[7] * a[8]) / a[10]
        out = cga_object(a, True)
        if np.linalg.norm(study_var(out)) > tol:
            continue
        return out


def rand_zero(maximum=10, tol=0):
    """Generates random rotor for zero displacement

    Parameters
    ----------
    maximum : int (float)
        Maximum for rational coefficients
         (Default value = 10)
    tol : float (optional)
        Tolerance for numerical error
         (Default value = 0)

    Returns
    -------
    (cga_object)
        Random rotor of zero displacement

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
