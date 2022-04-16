from .kinematic_polynomials import poly_act
from .geom_generators import point, point_to_cartesian
import numpy as np
import multiprocessing as mp
from functools import partial


def point_p_act_helper(i, points, param, poly):
    return point_to_cartesian(poly_act(param, poly, point(points[i])))


def point_p_act(points, param, poly):
    """Use poly_act on an array of points given in cartesian coordinates.

    Parameters
    ----------
    points : ndarray (N,3)
        Cartesian coordinates of points to act upon
    param: float
        parameter of polynomial
    poly: ndarray
        polynomial given as coefficient list with increasing degree

    Returns
    -------
    ndarray (N,3)
        Cartesian coordinates of points after rotor application

    """
    return np.array(
        mp.Pool().map(
            partial(point_p_act_helper, points=points, param=param, poly=poly),
            range(points.shape[0]),
        )
    )
