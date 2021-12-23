from .multivector import *
from .base_objects import *

def arr_to_quat(arr):
    """Converts an array to quaternion with coefficients given in array

    Args:
        lst (array): TODO

    Returns: cga_object

    """
    return arr[0] + q_i*arr[1] + q_j*arr[2] + q_k*arr[3]

def quat_to_arr(q):
    """gives coefficients of quaternion given in CGA as an array

    Args:
        q (cga_object): quaternion of qhich coefficients are to be given

    Returns: array

    """
    return np.array([q.coeff[0], -q.coeff[10], q.coeff[7], -q.coeff[6]])

def rotor_to_quat(rot):
    """Convert Rotor to its four quaternion representation

    Args:
        rot (cga_object):

    Returns: list of quaternions

    """
    coeff = rot.even_coeff
    p0  = coeff[0]  - q_i*coeff[5]  + q_j*coeff[2]  - q_k*coeff[1]
    q1  = coeff[11] + q_i*coeff[8]  + q_j*coeff[3]  - q_k*coeff[6]
    q2  = coeff[12] + q_i*coeff[9]  + q_j*coeff[4]  - q_k*coeff[7]
    q3  = coeff[10] - q_i*coeff[13] - q_j*coeff[15] - q_k*coeff[14]
    q0  = p0-q3
    return q0, q1, q2, q3

def quat_to_rotor(q0, q1, q2, q3):
    """Compute rotor given by the four quaternions q0,q1,q2,q3 as lists so that
    h = q0 + eps_1*q1 + eps_2*q2 + eps_3*q3

    Args:
        q0 (nd.array): Quaternion given in List form
        q1 (nd.array): Quaternion given in List form
        q2 (nd.array): Quaternion given in List form
        q3 (nd.array): Quaternion given in List form

    Returns: (cga_object) Rotor defined by the four quaternions

    """
    p0 = q0[1] + q_i*q0[2] + q_j*q0[3] + q_k*q0[4]
    p1 = q1[1] + q_i*q1[2] + q_j*q1[3] + q_k*q1[4]
    p2 = q2[1] + q_i*q2[2] + q_j*q2[3] + q_k*q2[4]
    p3 = q3[1] + q_i*q3[2] + q_j*q3[3] + q_k*q3[4]

    return p0 + eps_1*p1 + eps_2*p2 + eps_3*p3


