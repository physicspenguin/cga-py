from cga_py import *
import numpy as np
from scipy.optimize import root as solve


a = e_12 + 3 * e_1o
b = e_1 + e_12
s1 = sphere([0, 0, 0], 1)
s2 = sphere([1, 0, 0], 1)

q0 = arr_to_quat([1, 2, 3, 4])

point_p_act(np.array([[1, 1, 1], [-1, -1, -1]]), 4, [1, iso_scale([0, 0, 0])])
