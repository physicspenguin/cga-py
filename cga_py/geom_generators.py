from .multivector import *
from .base_objects import *

def point(x, y, z, conformal = True):
    """Generate Point in (x,y,z)

    Args:
        x (float): x-coordinate of point
        y (float): y-coordinate of point
        z (float): z-coordinate of point

    Kwargs:
        conformal (bool): return conformal description or euclidean

    Returns: cga_object

    """
    if conformal:
        return x*e_1 + y*e_2 + z*e_3 + 1/2*(x**2 + y**2 + z**2)*e_i + e_o
    else:
        return x*e_1 + y*e_2 + z*e_3

def sphere(center, radius):
    """Generate conformal representation of a sphere in center with radius

    Args:
        center (nd.array): center as [x,y,z]
        radius (float): radius of sphere

    Returns: cga_object

    """
    return (center[1]*e_1 + center[2]*e_2 + center[3]*e_3 +
            (1/2)*(center[1]**2 + center[2]**2 + center[3]**2 - radius**2)*e_i +
            e_o)

def plane(normal, distance):
    """Generate conformal representation of s plane with normalvector normal and
    distance to origin distance

    Args:
        normal (nd.array): Normal vector given as [x,y,z]
        distance (float): Distance to origin

    Returns: cga_object

    """
    c = normal/np.sqrt(normal[1]**2+normal[2]**2+normal[3]**2)
    return c[1]*e_1 + c[2]*e_2 + c[3]*e_3 + distance*e_i

def normalize_point(point):
    """normalize CGA representation of point

    Args:
        point (cga_object): Point in conformal representation

    Returns: cga_object

    """
    return cga_object(1/point.coeff[5]*point.coeff)

