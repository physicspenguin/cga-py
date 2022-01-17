from .multivector import *
from .base_objects import *
import cmath as cm

def point(point, conformal = True):
    """Generate Point in (x,y,z)

    Parameters
    ----------
    point : nd.array
        array of form [x,y,z] giving the coordinates of the point
        Kwargs:
    conformal : bool
        return conformal description or euclidean
        Returns: cga_object (Default value = True)

    Returns
    -------

    """
    if conformal:
        return (point[0]*e_1 + point[1]*e_2 + point[2]*e_3 +
                1/2*(point[0]**2 + point[1]**2 + point[2]**2)*e_i + e_o)
    else:
        return point[0]*e_1 + point[1]*e_2 + point[2]*e_3

def normalize_point(point):
    """normalize CGA representation of point

    Parameters
    ----------
    point : cga_object
        Point in conformal representation
        Returns: cga_object

    Returns
    -------

    """
    if any(sphere.coeff[6:-1]):
        print("Object is not a cga representation of a Point")
        return

    return cga_object(1/point.coeff[5]*point.coeff)

def point_to_cartesian(point):
    """Convert cga representation of a point into cartesian coordinates

    Parameters
    ----------
    point : cga_object
        Point to be converted into cartesian system
        Returns: (nd.array)

    Returns
    -------

    """
    if any(sphere.coeff[6:-1]):
        print("Object is not a cga representation of a Point")
        return
    return(np.array(normalize_point(point).coeff[1:4]))

def sphere(center, radius):
    """Generate conformal representation of a sphere in center with radius

    Parameters
    ----------
    center : nd.array
        center as [x,y,z]
    radius : float
        radius of sphere
        Returns: cga_object

    Returns
    -------

    """
    return (center[0]*e_1 + center[1]*e_2 + center[2]*e_3 +
            (1/2)*(center[0]**2 + center[1]**2 + center[2]**2 - radius**2)*e_i +
            e_o)

def normalize_sphere(sphere):
    """Normalize CGA representation of a sphere

    Parameters
    ----------
    sphere : cga_object
        cga representation of a sphere
        Returns: cga representation of the sphere normalized to e0

    Returns
    -------

    """
    if any(sphere.coeff[6:-1]):
        print("Object is not a cga representation of a sphere")
        return
    return cga_object(1/sphere.coeff[5]*sphere.coeff)

def sphere_to_cartesian(sphere):
    """Convert sphere in CGA representation to midpoint and radius

    Parameters
    ----------
    sphere : cga_object
        Sphere to be converted to cartesian
        Returns: (nd.array, float)

    Returns
    -------

    """
    if any(sphere.coeff[6:-1]):
        print("Object is not a cga representation of a sphere")
        return

    norm_sphere = normalize_sphere(sphere)
    x = norm_sphere.coeff[1]
    y = norm_sphere.coeff[2]
    z = norm_sphere.coeff[3]
    r = x**2 + y**2 + z**2 - 2*norm_sphere.coeff[4]
    if r>=0:
        return np.array([x, y, z]), np.sqrt(r)
    else:
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
    c = normal/np.sqrt(normal[0]**2+normal[1]**2+normal[2]**2)
    return c[0]*e_1 + c[1]*e_2 + c[2]*e_3 + distance*e_i

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
    if any(plane.coeff[5:-1]):
        print("Object is not a cga representation of a plane")
        return
    norm = 1/np.sqrt(plane.coeff[1]**2+plane.coeff[2]**2+plane.coeff[3]**2)
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
    if any(plane.coeff[5:-1]):
        print("Object is not a cga representation of a plane")
        return
    c = normalize_plane(plane).coeff
    return np.array([c[1], c[2], c[3]]), c[4]


