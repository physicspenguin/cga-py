from cga_py import *

import numpy as np
from scipy.optimize import root as solve
a = e_12 + 3*e_1o
b = e_1 + e_12
s1 = sphere([0,0,0], 1)
s2 = sphere([1,0,0], 1)

# print(sphere_to_cartesian(act(s1,s2)))
si = s2
for i in range(5):
    si = act(s1,si)
    print(si)
# print(sphere_to_cartesian(s))
# print(s)
# s = 2*s
# print(s)
# print(normalize_sphere(s))
# print(sphere_to_cartesian(s))
# p = point([1,2,3])
# print(p)
# p = 2*p
# print(p)
# print(normalize_point(p))
# print(point_to_cartesian(p))

# p = rand_point()
# print(p)
# print(point_to_cartesian(p))

# s = rand_sphere()
# print(s)
# print(sphere_to_cartesian(s))

# pl = plane([1,2,3],4)

# x = rand_rot_poly()
# print(x)
# print(study_var(x))

q0 = arr_to_quat([1,2,3,4])

point_p_act(np.array([[1,1,1],[-1,-1,-1]]), 4, [1,iso_scale([0,0,0])])
