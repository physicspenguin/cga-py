from .operators import act


def poly_act(param, poly, obj):
    """Let the motion polynomial poly(param) act on obj

    Parameters
    ----------
    param : float
        parameter of polynomial
    poly : ndarray
        polynomial represented by coefficients in list form starting at degree
        0 and increasing with array length
    obj : cga_object
        Object on which to let the rotor polynomial act

    Returns
    -------
    cga_object

    """
    return act(sum([poly[i] * (param**i) for i in range(len(poly))]), obj)
