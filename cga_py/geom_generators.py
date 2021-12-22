from .multivector import *
from .base_objects import *

def point(point, conformal = True):
    """Generate Point in (x,y,z)

    Args:
        point (nd.array): array of form [x,y,z] giving the coordinates of the point

    Kwargs:
        conformal (bool): return conformal description or euclidean

    Returns: cga_object

    """
    if conformal:
        return (point[0]*e_1 + point[1]*e_2 + point[2]*e_3 +
                1/2*(point[0]**2 + point[1]**2 + point[2]**2)*e_i + e_o)
    else:
        return point[0]*e_1 + point[1]*e_2 + point[2]*e_3

def normalize_point(point):
    """normalize CGA representation of point

    Args:
        point (cga_object): Point in conformal representation

    Returns: cga_object

    """
    return cga_object(1/point.coeff[5]*point.coeff)

def point_to_cartesian(point):
    """Convert cga representation of a point into cartesian coordinates

    Args:
        point (cga_object): Point to be converted into cartesian system

    Returns: (nd.array)

    """
    return(np.array(normalize_point(point).coeff[1:4]))

def sphere(center, radius):
    """Generate conformal representation of a sphere in center with radius

    Args:
        center (nd.array): center as [x,y,z]
        radius (float): radius of sphere

    Returns: cga_object

    """
    return (center[0]*e_1 + center[1]*e_2 + center[2]*e_3 +
            (1/2)*(center[0]**2 + center[1]**2 + center[2]**2 - radius**2)*e_i +
            e_o)

def normalize_sphere(sphere):
    """Normalize cga representation of a sphere

    Args:
        sphere (cga_object): cga representation of a sphere

    Returns: cga representation of the sphere normalized to e0

    """
    return cga_object(1/sphere.coeff[5]*sphere.coeff)

def sphere_to_cartesian(sphere):
    """TODO: Docstring for sphere_to_cartesian.

    Args:
        sphere (cga_object): Sphere to be converted to cartesian

    Returns: (nd.array, float)

    """
    norm_sphere = normalize_sphere(sphere)
    x = norm_sphere.coeff[1]
    y = norm_sphere.coeff[2]
    z = norm_sphere.coeff[3]
    r = np.sqrt(x**2 + y**2 + z**2 - 2*norm_sphere.coeff[4])

    return np.array([x, y, z]), r

def plane(normal, distance):
    """Generate conformal representation of s plane with normalvector normal and
    distance to origin distance

    Args:
        normal (nd.array): Normal vector given as [x,y,z]
        distance (float): Distance to origin

    Returns: cga_object

    """
    c = normal/np.sqrt(normal[0]**2+normal[1]**2+normal[2]**2)
    return c[0]*e_1 + c[1]*e_2 + c[2]*e_3 + distance*e_i




