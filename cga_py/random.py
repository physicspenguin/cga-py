import numpy.random as rand
from .multivector import *
from .base_objects import *
from .geom_generators import *

def rand_rational(maximum=10):
    """Generate randomn rational number up to n

    Kwargs:
        maximum (int): Maximum of random number generated

    Returns: TODO

    """
    return (-1)**(rand.randint(2))*(rand.randint(maximum) /
                                    (rand.randint(maximum)+1))

def rand_point(maximum = 10, conformal = True):
    """TODO: Generate random Point with rational coefficients

    Kwargs:
        maximum (int): Maximum of random number generated
        conformal (bool): Is conformal representation to be used

    Returns: cga_object

    """
    return point([rand_rational(maximum),
                 rand_rational(maximum),
                 rand_rational(maximum)],
                 conformal)

def rand_sphere(maximum = 10):
    """Generate random Sphere with rational parameters

    Kwargs:
        maximum (int): Maximum of rational parameters

    Returns: cga_object

    """
    return sphere([rand_rational(maximum),
                   rand_rational(maximum),
                   rand_rational(maximum)],
                  rand_rational(maximum))

def rand_plane(maximum = 10):
    """Generate random Plane with rational parameters

    Kwargs:
        maximum (int): Maximum of rational parameters

    Returns: cga_object

    """
    return plane([rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum)],
                 rand_rational(maximum))

def rand_rotor(maximum = 10):
    """Generates random rotor with rational coefficients

    Kwargs:
        maximum (float): maximum value

    Returns: TODO

    """
    pass

