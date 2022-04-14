from .kinematic_polynomials import poly_act
from .geom_generators import point, point_to_cartesian

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
    p = points.copy()
    for i in range(p.shape[0]):
        p[i] = point_to_cartesian(poly_act(param, poly, point(p[i])))
    return p

