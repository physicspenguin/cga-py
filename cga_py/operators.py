from .multivector import *


def act(rotator, obj):
    """Function to let rotator act on obj

    Parameters
    ----------
    rotator : cga_object
        rotator to apply to obj
    obj : cga_object
        Object to be acted upon
        Returns: (cga_object) Rotated object

    Returns
    -------

    """
    return rotator * obj * (~rotator)


def com(a, b):
    """commutator Product of a and b

    Parameters
    ----------
    a : cga_object
        TODO
    b : cga_object
        TODO
        Returns: cga_object

    Returns
    -------

    """
    return (1 / 2) * (a * b - b * a)


def anti_com(a, b):
    """anti-commutator Product of a and b

    Parameters
    ----------
    a : cga_object
        TODO
    b : cga_object
        TODO
        Returns: cga_object

    Returns
    -------

    """
    return (1 / 2) * (a * b + b * a)


def n_grade(obj, grade):
    """Returns parts of obj with grade grade

    Parameters
    ----------
    obj : cga_object
        object of which to extract grade-graded parts
    grade : int
        grade of parts to be extracted
        Returns: (cga_object) grade-graded parts of obj

    Returns
    -------

    """
    vec = np.zeros(32)
    if grade == 0:
        ind = 0
        vec[ind] = obj.coeff[ind]
        return cga_object(vec)
    elif grade == 1:
        ind = range(1, 6)
        vec[ind] = obj.coeff[ind]
        return cga_object(vec)
    elif grade == 2:
        ind = range(6, 16)
        vec[ind] = obj.coeff[ind]
        return cga_object(vec)
    elif grade == 3:
        ind = range(16, 26)
        vec[ind] = obj.coeff[ind]
        return cga_object(vec)
    elif grade == 4:
        ind = range(26, 31)
        vec[ind] = obj.coeff[ind]
        return cga_object(vec)
    else:
        ind = 31
        vec[ind] = obj.coeff[ind]
        return cga_object(vec)


def r_norm(obj):
    """Return obj*(obj^*) in CGA

    Parameters
    ----------
    obj :
        cga_object

    Returns
    -------
    type


    """
    return obj * (~obj)


def l_norm(obj):
    """Return (obj^*)*obj in CGA

    Parameters
    ----------
    obj :
        cga_object

    Returns
    -------
    type


    """
    return (~obj) * obj
