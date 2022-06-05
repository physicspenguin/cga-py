from .multivector import cga_object
from .base_objects import e_1, e_2, e_3, e_i, e_o
import numpy as np
import cmath as cm


def point(point):
    """Generate Point in (x,y,z)

    Parameters
    ----------
    point : array_like
        Cartesian coordinates of point given as [x, y, z]

    Returns
    -------
    cga_object
        CGA representation of point with coordinates `point`

    Notes
    -----
    In CGA all objects can be interpreted as spheres given by:
        x*e_1 + y*e_2 + z*e_3 + 1/2*(x^2 + y^2 + z^2 - r^2)*e_i + e_0
    Thus a point can be seen as a sphere with radius r = 0 with center
    [x, y, z].
    So a point is represented as:
        x*e_1 + y*e_2 + z*e_3 + 1/2*(x^2 + y^2 + z^2)*e_i + e_0

    """
    if len(point) != 3:
        raise ValueError("Point must have three coordinates.")
    return (
        point[0] * e_1
        + point[1] * e_2
        + point[2] * e_3
        + 1 / 2 * (point[0] ** 2 + point[1] ** 2 + point[2] ** 2) * e_i
        + e_o
    )


def normalize_point(point):
    """Normalize CGA representation of point

    Parameters
    ----------
    point : cga_object
        Point in conformal representation.

    Returns
    -------
    cga_object
        `point` given in its normalized CGA representation.

    """
    num_max = max(abs(point.coeff[1:5]))
    if not np.isclose(point.coeff[0] / num_max, 0):
        raise ValueError(str(point) + " is not a cga representation of a point")

    if not np.allclose(point.coeff[6:-1] / num_max, np.zeros_like(point.coeff[6:-1])):
        raise ValueError(str(point) + " is not a cga representation of a point")

    return cga_object(1 / point.coeff[5] * point.coeff)


def point_to_cartesian(point):
    """Convert CGA representation of a point into cartesian coordinates.

    Parameters
    ----------
    point : cga_object
        Point to be converted into cartesian system.

    Returns
    -------
    array_like
        Cartesian coordinates of `point`.

    """
    return np.array(normalize_point(point).coeff[1:4], dtype=complex)


def sphere(center, radius):
    """Generate conformal representation of a sphere in `center` with `radius`

    Parameters
    ----------
    center : array_like
        Center of sphere as [x,y,z].
    radius : float
        Radius of sphere.

    Returns
    -------
    cga_object
        CGA representation of Sphere with center in `center` and radius
        `radius`.

    Notes
    -----
    In CGA all objects can be interpreted as spheres given by:
        x*e_1 + y*e_2 + z*e_3 + 1/2*(x^2 + y^2 + z^2 - r^2)*e_i + e_0

    """
    if len(center) != 3:
        raise ValueError("Center must have three coordinates.")
    return (
        center[0] * e_1
        + center[1] * e_2
        + center[2] * e_3
        + (1 / 2)
        * (center[0] ** 2 + center[1] ** 2 + center[2] ** 2 - radius**2)
        * e_i
        + e_o
    )


def normalize_sphere(sphere):
    """Normalize CGA representation of sphere.

    Parameters
    ----------
    sphere : cga_object
        CGA representation of a sphere.

    Returns
    -------
    cga_object
        CGA representation of the sphere normalized to e0.


    """
    num_max = max(abs(sphere.coeff[1:5]))
    if not np.isclose(sphere.coeff[0] / num_max, 0):
        raise ValueError("Object is not a cga representation of a sphere")

    if not np.allclose(sphere.coeff[6:-1] / num_max, np.zeros_like(sphere.coeff[6:-1])):
        raise ValueError("Object is not a cga representation of a sphere")
    return cga_object(1 / sphere.coeff[5] * sphere.coeff)


def sphere_to_cartesian(sphere):
    """Convert sphere in CGA representation to center and radius.

    Parameters
    ----------
    sphere : cga_object
        Sphere to be converted to cartesian.

    Returns
    -------
    (array_like, float)
        Center of sphere given as [x, y, z] and radius of sphere.

    """
    norm_sphere = normalize_sphere(sphere)
    x = norm_sphere.coeff[1]
    y = norm_sphere.coeff[2]
    z = norm_sphere.coeff[3]
    r = x**2 + y**2 + z**2 - 2 * norm_sphere.coeff[4]
    return np.array([x, y, z]), cm.sqrt(r)


def plane(normal, distance):
    """Generate conformal representation of s plane with normalvector normal and
    distance to origin distance

    Parameters
    ----------
    normal : nd.array
        Normal vector given as [x,y,z]
    distance : float
        Distance to origin
        Returns: cga_object

    Returns
    -------

    """
    if len(normal) != 3:
        raise ValueError("Normal must have three coordinates.")
    c = normal / np.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    return c[0] * e_1 + c[1] * e_2 + c[2] * e_3 + distance * e_i


def normalize_plane(plane):
    """Normalize CGA representation of a plane

    Parameters
    ----------
    plane : cga_object
        Plane to be normalized
        Returns: (cga_object)

    Returns
    -------

    """
    num_max = max(abs(plane.coeff[1:5]))
    if not np.isclose(plane.coeff[0] / num_max, 0):
        raise ValueError("Object is not a cga representation of a plane")

    if not np.allclose(plane.coeff[6:-1] / num_max, np.zeros_like(plane.coeff[6:-1])):
        raise ValueError("Object is not a cga representation of a plane")

    norm = 1 / np.sqrt(plane.coeff[1] ** 2 + plane.coeff[2] ** 2 + plane.coeff[3] ** 2)
    return cga_object(norm * plane.coeff)


def plane_to_cartesian(plane):
    """Convert plane given in CGA to representation ad normal vector and
    distance from origin.

    Parameters
    ----------
    plane : cga_object
        plane to be converted
        Returns: (nd.array, float) normal unit-vector, distance from origin

    Returns
    -------

    """
    c = normalize_plane(plane).coeff
    return np.array([c[1], c[2], c[3]]), c[4]
