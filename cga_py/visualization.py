from .kinematic_polynomials import poly_act
from .geom_generators import point, point_to_cartesian
import numpy as np
import multiprocessing as mp
from functools import partial
from numba import njit
import pyqtgraph.opengl as gl


def point_p_act_helper(i, points, param, poly):
    return point_to_cartesian(poly_act(param, poly, point(points[i])))


def point_p_act(points, param, poly):
    """Use poly_act on an array of points given in cartesian coordinates.

    Parameters
    ----------
    points : ndarray (N,3)
        Cartesian coordinates of points to act upon
    param: float
        parameter of polynomial
    poly: ndarray
        polynomial given as coefficient list with increasing degree

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
    pointsx = np.linspace(0, length[0], subd[0])
    pointsy = np.linspace(0, length[1], subd[1])
    pointsz = np.linspace(0, length[2], subd[2])

    return point_cube_gen_help(pointsx, pointsy, pointsz, center, length, subd)


@njit(parallel=True, cache=True)
def point_cube_gen_help(pointsx, pointsy, pointsz, center, length, subd):
    """TODO: Docstring for point_cube_gen_help.
    Returns
    -------
    TODO

    """
    lx = subd[0]
    ly = subd[1]
    lz = subd[2]
    plot_points = np.zeros((lx * ly * lz, 3))
    colors = np.ones((lx * ly * lz, 4))
    center_off = np.array(
        [
            center[0] - length[0] / 2,
            center[1] - length[1] / 2,
            center[2] - length[2] / 2,
        ]
    )
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                plot_points[lz * ly * x + lz * y + z, 0] = pointsx[x]
                plot_points[lz * ly * x + lz * y + z, 1] = pointsy[y]
                plot_points[lz * ly * x + lz * y + z, 2] = pointsz[z]
                colors[lz * ly * x + lz * y + z, 0] = pointsx[x] / length[0]
                colors[lz * ly * x + lz * y + z, 1] = pointsy[y] / length[1]
                colors[lz * ly * x + lz * y + z, 2] = pointsz[z] / length[2]
    return (plot_points + center_off), colors


def sphere_gen(
    center=np.array([0, 0, 0]),
    radius=1,
    rows=10,
    cols=20,
    color=(0.5, 0.5, 0.5, 0.5),
    smooth=True,
):
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
