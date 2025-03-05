from .multivector import cga_object
from .permutations_pm import (
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
from .base_objects import e_1, e_2, e_3, e_p, e_m
import numpy as np


def act(rotator, obj):
    """Function to let rotator act on obj

    Parameters
    ----------
    rotator : cga_object
        rotator to apply to obj
    obj : cga_object
        Object to be acted upon


    Returns
    -------
    cga_object
        Rotated object

    """
    if not isinstance(rotator, cga_object):
        rotator = cga_object(rotator)
    return rotator * obj * (~rotator)


def com(a, b):
    """commutator Product of a and b

    Parameters
    ----------
    a : cga_object
    b : cga_object

    Returns
    -------
    cga_object

    """
    return (1 / 2) * (a * b - b * a)


def anti_com(a, b):
    """anti-commutator Product of a and b

    Parameters
    ----------
    a : cga_object
    b : cga_object

    Returns
    -------
    cga_object

    """
    return (1 / 2) * (a * b + b * a)


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
    h = obj.coeff
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


def n_grade(obj, grade):
    """Returns parts of obj with grade grade

    Parameters
    ----------
    obj : cga_object
        object of which to extract grade-graded parts
    grade : int
        grade of parts to be extracted

    Returns
    -------
    cga_object
        grade-graded parts of obj

    """
    vec = np.zeros_like(obj.coeff)
    grade_vec = io_to_pm_vec(obj)
    if grade == 0:
        vec[0] = grade_vec[0]
        return pm_vec_to_obj(vec)
    elif grade == 1:
        vec[1:6] = grade_vec[1:6]
        return pm_vec_to_obj(vec)
    elif grade == 2:
        vec[6:16] = grade_vec[6:16]
        return pm_vec_to_obj(vec)
    elif grade == 3:
        vec[16:26] = grade_vec[16:26]
        return pm_vec_to_obj(vec)
    elif grade == 4:
        vec[26:31] = grade_vec[26:31]
        return pm_vec_to_obj(vec)
    else:
        vec[31] = grade_vec[31]
        return pm_vec_to_obj(vec)


def r_norm(obj):
    """Calculates the right norm in CGA

    Parameters
    ----------
    obj :
        cga_object

    Returns
    -------
    cga_object


    """
    return obj * (~obj)


def l_norm(obj):
    """Calculates the left norm in CGA

    Parameters
    ----------
    obj :
        cga_object

    Returns
    -------
    cga_object


    """
    return (~obj) * obj
