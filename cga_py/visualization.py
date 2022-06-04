from .kinematic_polynomials import poly_act
from .geom_generators import point, point_to_cartesian
import numpy as np
import multiprocessing as mp
from functools import partial
from numba import njit
import pyqtgraph.opengl as gl


def point_p_act_helper(i, param, poly, points):
    return point_to_cartesian(poly_act(param, poly, point(points[i])))


def point_p_act(param, poly, points):
    """Use poly_act on an array of points given in cartesian coordinates.

    Parameters
    ----------
    param: float
        parameter of polynomial
    poly: ndarray
        polynomial given as coefficient list with increasing degree
    points : ndarray (N,3)
        Cartesian coordinates of points to act upon

    Returns
    -------
    ndarray (N,3)
        Cartesian coordinates of points after rotor application

    """
    return np.array(
        mp.Pool().map(
            partial(point_p_act_helper, points=points, param=param, poly=poly),
            range(points.shape[0]),
        )
    )


def point_cube_gen(
    center=np.array([0, 0, 0]), length=np.array([1, 1, 1]), subd=np.array([10, 10, 10])
):
    """Generate cube wtih

    Parameters
    ----------
    length : nparray, optional
        Edge lengths of cube.
    center : nparray, optional
        Center of cube.
    subd : nparray, optional
        subdivisions along axes.

    Returns
    -------
    points: nparray(N,3)
        Array of points generating cube.
    colors: nparray(N,4)
        Array of colors for plotting.

    """
    subd = np.array(subd, dtype=int)
    center = np.array(center)
    length = np.array(length)
    len_div = np.array(length)
    for i in range(3):
        if subd[i] == 1:
            length[i] = 0
        if length[i] == 0:
            len_div[i] = 1
    pointsx = np.linspace(0, length[0], subd[0])
    pointsy = np.linspace(0, length[1], subd[1])
    pointsz = np.linspace(0, length[2], subd[2])

    return point_cube_gen_help(pointsx, pointsy, pointsz, center, length, len_div, subd)


@njit(parallel=True, cache=True)
def point_cube_gen_help(pointsx, pointsy, pointsz, center, length, len_div, subd):
    """TODO: Docstring for point_cube_gen_help.
    Returns
    -------
    TODO

    """
    sx = subd[0]
    sy = subd[1]
    sz = subd[2]
    # set length to 0 if there is only one subdivision to assure correct center
    plot_points = np.zeros((sx * sy * sz, 3))
    colors = np.ones((sx * sy * sz, 4))
    center_off = np.array(
        [
            center[0] - length[0] / 2,
            center[1] - length[1] / 2,
            center[2] - length[2] / 2,
        ]
    )
    # set length to 1 if it is zero to avoid division by zero
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                plot_points[sz * sy * x + sz * y + z, 0] = pointsx[x]
                plot_points[sz * sy * x + sz * y + z, 1] = pointsy[y]
                plot_points[sz * sy * x + sz * y + z, 2] = pointsz[z]
                colors[sz * sy * x + sz * y + z, 0] = 1 - (pointsx[x] / len_div[0])
                colors[sz * sy * x + sz * y + z, 1] = 1 - (pointsy[y] / len_div[1])
                colors[sz * sy * x + sz * y + z, 2] = 1 - (pointsz[z] / len_div[2])
    return (plot_points + center_off), colors


def sphere_gen(
    center=np.array([0, 0, 0]),
    radius=1,
    rows=10,
    cols=20,
    color=(0.5, 0.5, 0.5, 0.5),
    smooth=True,
):  # pragma: no cover
    """TODO: Docstring for sphere_gen.

    Parameters
    ----------
    center : ndarray (3), optional
        Center of sphere
    radius : float, optional
        Radius of sphere
    rows : int, optional
        Number of rows in sphere
    cols : int, optional
        Number of columns in sphere
    color: tuple (4), optional
        Default face colors
    smooth : boolean, optional
        Sould be shaded smooth

    Returns
    -------
    GLMeshItem
        Mesh of sphere


    """
    md = gl.MeshData.sphere(rows=rows, cols=cols)
    colors = np.ones((md.faceCount(), 4), dtype=float)
    md.setFaceColors(colors)
    m3 = gl.GLMeshItem(meshdata=md, smooth=smooth, shader="balloon")
    m3.translate(center[0], center[1], center[2])
    m3.scale(radius, radius, radius)
    m3.setColor(color)
    return m3
