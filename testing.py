from cga_py import *

import numpy as np
from scipy.optimize import root as solve
a = e_12 + 3*e_1o
b = e_1 + e_12
# print(rand_rotor())
# s = sphere([3,2,1], 4)
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
i = 0
while True:
    x = rand_zero()
    if np.linalg.norm(study_var(x))!=0:
        break
    if np.linalg.norm(null_quadric(x))!=0:
        break
    i += 1
    print("Successful Object generation number \t", i)

rot = cga_object([-10/21, -1/4, 4/17, -2, 2/5, 3/2, -21/4, 12/11, 9/7, 6, -5,
                  2979/680, 45423/18700, -6153/2200, -10119/425, -2934/55],
                 True)
nul = cga_object([1/2, -7/8, -1/2, -10/7, 1/2, 4/5, 19/5, -1/5, -19/2, 8/3,
                  7834493/256116, 5079/280, -61/15, -257669479/5122320,
                  -8316235/256116, 693151/21343],
                 True)
# study_var(rot)

y = rand_zero()

