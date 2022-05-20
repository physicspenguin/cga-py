import numpy as np
from .errors import GeometryError


def study_cond(a, b):
    """Computes Study condition for objects a and b embeded into the CGA

    Parameters
    ----------
    a : cga_object
        First parameter
    b : cga_object

    Returns
    -------
    cga_object
        Study condition of `a` and `b`
    """
    return a * (~b) + b * (~a)


def study_var(r):
    """Compute Study variety equations for even graded elements of CGA

    Parameters
    ----------
    r : cga_object
        even graded cga_object

    Returns
    -------
    ndarray

    """
    b = r.get_even()
    return np.array(
        [
            b[0] * b[11] - b[1] * b[8] + b[2] * b[6] - b[3] * b[5],
            b[0] * b[12] - b[1] * b[9] + b[2] * b[7] - b[4] * b[5],
            b[0] * b[13] - b[1] * b[10] + b[3] * b[7] - b[4] * b[6],
            b[0] * b[14] - b[2] * b[10] + b[3] * b[9] - b[4] * b[8],
            b[0] * b[15] - b[5] * b[10] + b[6] * b[9] - b[7] * b[8],
            -b[1] * b[14] + b[2] * b[13] - b[3] * b[12] + b[4] * b[11],
            b[1] * b[15] - b[5] * b[13] + b[6] * b[12] - b[7] * b[11],
            -b[2] * b[15] + b[5] * b[14] - b[8] * b[12] + b[9] * b[11],
            b[3] * b[15] - b[6] * b[14] + b[8] * b[13] - b[10] * b[11],
            -b[4] * b[15] + b[7] * b[14] - b[9] * b[13] + b[10] * b[12],
        ]
    )


def null_quadric(r):
    """Evaluates Nullquadric condition for even graded element r.
    Condition is true, if output is 0.

    Parameters
    ----------
    r : cga_object
        object of which to evaluate the null quadric condition

    Returns
    -------
    float

    """
    if r == r.make_even():
        return (r * (~r)).coeff[0]
    else:
        raise GeometryError("Element must be of even grade")
