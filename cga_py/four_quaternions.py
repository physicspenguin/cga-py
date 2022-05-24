from .multivector import cga_object
from .base_objects import q_i, q_j, q_k, eps_1, eps_2, eps_3
import numpy as np


def arr_to_quat(arr):
    """Converts an array to quaternion with coefficients given in array

    Parameters
    ----------
    arr : array_like
        array giving the coefficients of [1,i,j,k] in this order.

    Returns
    -------
    cga_object
        Quaternion given by the coefficients in `arr` represented in terms of
        CGA.

    """
    return arr[0] + q_i * arr[1] + q_j * arr[2] + q_k * arr[3]


def quat_to_arr(q):
    """Coefficients of quaternion `q` given in terms of CGA.

    Parameters
    ----------
    q : cga_object
        Quaternion of which coefficients are to be calculated.

    Returns
    -------
    array
        Coefficients of quaternion `q` given in order of [1,i,j,k].

    """
    return np.array([q.coeff[0], -q.coeff[10], q.coeff[7], -q.coeff[6]])


def rotor_to_quat(rot):
    """Convert Rotor to its four quaternion representation

    Parameters
    ----------
    rot : cga_object
        Returns: list of quaternions

    Returns
    -------
    tuple
        Tuple of quaternions that represent the rotor in terms of the four
        quaternion decomposition.

    Notes
    -----
    Rotors in CGA can be represented through four quaternions in the form
    q0 + eps_1*q1 + eps_2*q2 + eps_3*q3.
    These quaternions can be directly extracted from the CGA representation of
    the rotor.

    """
    coeff = rot.get_even()
    q0 = (
        (coeff[0] - coeff[10])
        + q_i * (coeff[15] - coeff[5])
        + q_j * (coeff[2] - coeff[14])
        + q_k * (coeff[13] - coeff[1])
    )
    q1 = coeff[11] + q_i * coeff[3] + q_j * coeff[6] + q_k * coeff[8]
    q2 = coeff[12] + q_i * coeff[4] + q_j * coeff[7] + q_k * coeff[9]
    q3 = coeff[10] + q_i * (-coeff[15]) + q_j * coeff[14] + q_k * (-coeff[13])
    return q0, q1, q2, q3


def quat_to_rotor(q0, q1, q2, q3):
    """Compute rotor given by the four quaternions q0,q1,q2,q3

    Parameters
    ----------
    q0 : array_like
        Quaternion given in list-form
    q1 : array_like
        Quaternion given in list-form
    q2 : array_like
        Quaternion given in list-form
    q3 : array_like
        Quaternion given in list-form

    Returns
    -------
    cga_object
        Rotor defined by the four quaternions.

    Notes
    -----
    Rotors can be represented by four quaternionsq0 q1,q2,q3 as lists so that
    h = q0 + eps_1*q1 + eps_2*q2 + eps_3*q3


    """
    if not (
        isinstance(q0, cga_object)
        and isinstance(q1, cga_object)
        and isinstance(q2, cga_object)
        and isinstance(q0, cga_object)
    ):
        p0 = q0[0] + q_i * q0[1] + q_j * q0[2] + q_k * q0[3]
        p1 = q1[0] + q_i * q1[1] + q_j * q1[2] + q_k * q1[3]
        p2 = q2[0] + q_i * q2[1] + q_j * q2[2] + q_k * q2[3]
        p3 = q3[0] + q_i * q3[1] + q_j * q3[2] + q_k * q3[3]
    else:
        p0 = q0
        p1 = q1
        p2 = q2
        p3 = q3

    return p0 + eps_1 * p1 + eps_2 * p2 + eps_3 * p3


def vectorial(quat):
    """Returns vectorial part of quaternion

    Parameters
    ----------
    quat : cga_object or array_like
        Quaternion of which to to extract the vectorial component.

    Returns
    -------
    cga_object or array_like
        Vectorial part of quaternion in same representation as input datatype.

    """
    if isinstance(quat, cga_object):
        return 1 / 2 * (quat - (~quat))
    else:
        return quat[1:-1]


def scalar(quat):
    """Returns scalar part of quaternion

    Parameters
    ----------
    quat : cga_object or array_like
        Quaternion of which to extract the scalar component.

    Returns
    -------
    cga_object or array_like
        Vectorial part of quaternion in same representation as input datatype.

    """
    if isinstance(quat, cga_object):
        return quat - 1 / 2 * (quat - (~quat))
    else:
        return quat[0]
