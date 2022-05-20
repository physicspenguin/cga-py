import numpy as np

# Regular Import of the CGA library
import cga_py as cg

# Import of standard objects such as e_1, e_i, eps_1 and such.
from cga_py.base_objects import *

# Import of all permutations of indices of standard objects.
# Not necessary, but a quality of life improvement.
from cga_py.permutations import *

########################################
# General object creation
########################################

# Genetare CGA-object through definition of all indices.
# Array of length upto 32 can be used. Undefined parameters are filled with 0.
a = cg.cga_object(np.arange(32))
print(a)

# Genetare CGA-object through definition of even indices.
# Array of length upto 16 can be used. Undefined parameters are filled with 0.
b = cg.cga_object(np.arange(16), True)
print(b)

# Genetare CGA-object through sum of CGA-Objects
# Array of length upto 16 can be used. Undefined parameters are filled with 0.
r = cg.cga_object(e_12 + 5 * e_1)
print(r)

########################################
# General Arithmetic
########################################

# Sums
# Sum of two CGA-Objects
print(e_1 + e_i)
# Sum of CGA-object and numeric type
print(1 + e_2)

# Multiplication
# Product of two CGA-Objects
print(e_1 * e_12)
# Product of CGA-object and numeric type
print(5 * e_1)

# Division of CGA-object by numeric type
print(b / 5)

# Floordivision of CGA-object by numeric type
print(b // 5)

# Wedge product
print(e_1 ^ e_2)

# Scalar product
print(e_1 | e_2)

# Reversion
print(~e_12)


########################################
# Specialized methods
########################################

# Get coefficients of CGA-object
print(b.coeff)

# Get even part of CGA-object
print(a.get_even())

# Generate even graded-object of a specified CGA-object
print(a.make_even())


########################################
# Operators
########################################

# Let a rotator r act upon CGA-object a
print(cg.act(r, a))

# Commutator product of a and b
print(cg.com(a, b))

# Anti-commutator product of a and b
print(cg.anti_com(a, b))

# Extract parts with grade n of CGA-object a
print(cg.n_grade(a, 3))

# Calculate right norm of a
print(cg.r_norm(a))

# Calculate left norm of a
print(cg.l_norm(a))


########################################
# Checking specific conditions
########################################

# Study condition of a and b
print(cg.study_cond(a, b))

# Evaluate Study variety condition for a
print(cg.study_var(a))

# Evaluate null quadric condition for a
print(cg.null_quadric(a))


########################################
# Geometric objects in CGA
########################################

# Points
# Generate point
print(cg.point([1, 2, 3]))
# Normalize point
print(cg.normalize_point(5 * cg.point([1, 2, 3])))
# Convert point to cartesian form
print(cg.point_to_cartesian(5 * cg.point([1, 2, 3])))

# Planes
# Generate plane
print(cg.plane([1, 2, 3], 2))
# Normalize plane
print(cg.normalize_plane(5 * cg.plane([1, 2, 3], 2)))
# Convert plane to cartesian form
print(cg.plane_to_cartesian(5 * cg.plane([1, 2, 3], 2)))

# Spheres
# Generate sphere
print(cg.sphere([1, 2, 3], 2))
# Normalize sphere
print(cg.normalize_sphere(5 * cg.sphere([1, 2, 3], 2)))
# Convert sphere to cartesian form
print(cg.sphere_to_cartesian(5 * cg.sphere([1, 2, 3], 2)))


########################################
# Generate random objects
########################################

# Generates a random rational number
print(cg.rand_rational())

# Generates a random point in CGA representation
print(cg.rand_point())

# Generates a random plane in CGA representation
print(cg.rand_plane())

# Generates a random sphere in CGA representation
print(cg.rand_sphere())

# Generates a random rotor that can be applied to a CGA-object
print(cg.rand_rotor())

# Generates a random rotor polynomial,that can be applied to a CGA-object
print(cg.rand_rot_poly())

# Generates a random rotor with zero displacement,
# that can be applied to a CGA-object
print(cg.rand_zero())


########################################
# Four quaternion representation
########################################

# Convert array of coefficients to quaternion in CGA representation
print(cg.arr_to_quat([1, 2, 3, 4]))

# Convert quaternion in CGA representation to array of coefficients
print(cg.quat_to_arr(1 + 2 * q_i + 3 * q_j + 4 * q_k))

# Convert rotor to four quaternion representation
print(cg.rotor_to_quat(e_12 + 5 * e_1))

# Convert four quaternions to rotor
print(cg.quat_to_rotor(1, q_i, q_j, q_k))

# Extract vectorial part of a quaternion
print(cg.vectorial(1 + 2 * q_i + 3 * q_j + 4 * q_k))
