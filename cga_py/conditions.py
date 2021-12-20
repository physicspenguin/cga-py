from .multivector import *
from .base_objects import *

def study_cond(a,b):
    """Computes Study condition for quaternions a and b embeded into the CGA

    Args:
        a (cga_object):
        b (cga_object):

    Returns: cga_object

    """
    return a*(~b) + b*(~a)

def study_var(r):
    """Compute Study variety equations for even graded elements of CGA

    Args:
        r (cga_object): even graded cga_object

    Returns: nd.array

    """
    a1 = r.coeff[1]
    a2 = r.coeff[6]
    a3 = r.coeff[7]
    a4 = r.coeff[10]
    a5 = r.coeff[8]
    a6 = r.coeff[11]
    a7 = r.coeff[13]
    a8 = r.coeff[26]
    c1 = r.coeff[28]
    c2 = r.coeff[15]
    c3 = r.coeff[30]
    c4 = r.coeff[29]
    c5 = r.coeff[12]
    c6 = r.coeff[9]
    c7 = r.coeff[27]
    c8 = r.coeff[14]

    return np.array([a1*a8 - a2*a7 + a3*a6 - a4*a5, a1*c7 - a2*c8 + a3*c5 -
                     a4*c6, a1*c1 - a2*c2 + a5*c5 - a6*c6, a1*c4 - a3*c2 +
                     a5*c8 - a7*c6, a1*c3 - a4*c2 + a6*c8 - a7*c5, -a2*c4 +
                     a3*c1 - a5*c7 + a8*c6, a2*c3 - a4*c1 + a6*c7 - a8*c5,
                     -a3*c3 + a4*c4 - a7*c7 + a8*c8, a5*c3 - a6*c4 + a7*c1 -
                     a8*c2, -c1*c8 + c2*c7 - c3*c6 + c4*c5])

def null_quadric(r):
    """Evaluates Nullquadric condition for r. Condition is true, if output is 0.

    Args:
        r (cga_object): object of which to evaluate the null quadric condition

    Returns: float

    """
    return (r*(~r)).coeff[0]


