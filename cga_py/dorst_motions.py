from .base_objects import e_1, e_2, e_3, e_i
from .multivector import cga_object
from .geom_generators import point_to_cartesian, point
from .errors import GeometryError


def iso_scale(a):
    """Generates the rotor for an isotropic scaling with center a

    Parameters
    ----------
    a : cga_object, ndarray
        scaling center as point in CGA representation

    Returns
    -------
    cga_object
        rotor corresponding to a uniform scaling with center a

    """
    if not isinstance(a, cga_object):
        a = point(a)
    return a ^ e_i


def transv(a, b):
    """Generates transversion defined by the point a and the orthogonal plane b

    Parameters
    ----------
    a : cga_object, ndarray
        Point a given as cga_object or as list of coordinates
    b : cga_object, ndarray
        Plane given as cga_object or as list of coordinates for its normal-vector
    Returns
    -------
    cga_object
        rotor corresponding to a transversion defined by a and b.

    """
    # Check if a is given in CGA representation or as coordinates
    if isinstance(a, cga_object):
        coord = point_to_cartesian(a)
    else:
        # Extract coordinates and generate the CGA representation
        coord = a
        a = point(coord)
    if isinstance(b, cga_object):
        if (a | b) != cga_object([0]):
            raise GeometryError("Plane b must be orthogonal to point a")
    else:
        b = (
            b[0] * e_1
            + b[1] * e_2
            + b[2] * e_3
            + (coord[0] * b[0] + coord[1] * b[1] + coord[2] * b[2]) * e_i
        )

    return a ^ b
