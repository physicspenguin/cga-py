import numpy.random as rand
from scipy.optimize import curve_fit as solve
from .multivector import *
from .base_objects import *
from .geom_generators import *
from .conditions import *

def rand_rational(maximum=10):
    """Generate random rational number up to n

    Kwargs:
        maximum (int): Maximum of random number generated

    Returns: (float) Random rational number in Range maximum

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
        maximum (int): maximum for rational coefficients

    Returns: cga-object

    """
    x = np.array([rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum)])
    study = lambda y,x12,x13,x14,x15,x16: study_var(cga_object(
        np.append(x,np.array([x12,x13,x14,x15,x16])),True))
    return solve(study,np.array([0,0,0,0,0]))

def rand_rot_poly(maximum = 10):
    """Generates random rotor for rotor polynomials with rational coefficients.

    Kwargs:
        maximum (int): maximum for rational coefficients

    Returns: cga-object

    """
    pass

def rand_zero(maximum = 10):
    """Generates random rotor for zero displacement

    Kwargs:
        maximum (int): maximum for rational coefficients

    Returns: cga-object

    """
    pass

