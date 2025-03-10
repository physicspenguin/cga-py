from .multivector import cga_object
import numpy as np

# Define Base Objects defining the Standard representation of vectors in CGA
# based upon the viewing of the R3 as a 4 dimensional projective space.

# Base objects of grade 1
e_1 = cga_object(np.eye(32)[:, 1])
e_2 = cga_object(np.eye(32)[:, 2])
e_3 = cga_object(np.eye(32)[:, 3])
e_i = cga_object(np.eye(32)[:, 4])
e_o = cga_object(np.eye(32)[:, 5])

# Base objects of grade 2
e_12 = cga_object(np.eye(32)[:, 6])
e_13 = cga_object(np.eye(32)[:, 7])
e_1i = cga_object(np.eye(32)[:, 8])
e_1o = cga_object(np.eye(32)[:, 9])
e_23 = cga_object(np.eye(32)[:, 10])
e_2i = cga_object(np.eye(32)[:, 11])
e_2o = cga_object(np.eye(32)[:, 12])
e_3i = cga_object(np.eye(32)[:, 13])
e_3o = cga_object(np.eye(32)[:, 14])
e_io = cga_object(np.eye(32)[:, 15])

# Base objects of grade 3
e_123 = cga_object(np.eye(32)[:, 16])
e_12i = cga_object(np.eye(32)[:, 17])
e_12o = cga_object(np.eye(32)[:, 18])
e_13i = cga_object(np.eye(32)[:, 19])
e_13o = cga_object(np.eye(32)[:, 20])
e_1io = cga_object(np.eye(32)[:, 21])
e_23i = cga_object(np.eye(32)[:, 22])
e_23o = cga_object(np.eye(32)[:, 23])
e_2io = cga_object(np.eye(32)[:, 24])
e_3io = cga_object(np.eye(32)[:, 25])

# Base objects of grade 4
e_123i = cga_object(np.eye(32)[:, 26])
e_123o = cga_object(np.eye(32)[:, 27])
e_12io = cga_object(np.eye(32)[:, 28])
e_13io = cga_object(np.eye(32)[:, 29])
e_23io = cga_object(np.eye(32)[:, 30])

# Base objects of grade 5
e_123io = cga_object(np.eye(32)[:, 31])

# Definition of epsilon 1, 2 and 3 for representation of CGA as four
# quaternions.
eps_1 = e_123i
eps_2 = e_123o
eps_3 = eps_1 * eps_2 + 1

# Embedding of quaternions in the CGA
q_i = -e_23
q_j = e_13
q_k = -e_12


# Plus Minus Basis
e_p = 1 / 2 * e_i - e_o
e_m = 1 / 2 * e_i + e_o

# e_1p = e_1 * e_p
# e_1m = e_1 * e_m
# e_2p = e_2 * e_p
# e_2m = e_2 * e_m
# e_3p = e_3 * e_p
# e_3m = e_3 * e_m
# e_pm = e_p * e_m
#
# e_12p = e_12 * e_p
# e_12m = e_12 * e_m
# e_13p = e_13 * e_p
# e_13m = e_13 * e_m
# e_1pm = e_1p * e_m
# e_23p = e_23 * e_p
# e_23m = e_23 * e_m
# e_2pm = e_2p * e_m
# e_3pm = e_3p * e_m
#
# # Base mbjects mf grade 4
# e_123p = e_123 * e_p
# e_123m = e_123 * e_m
# e_12pm = e_12p * e_m
# e_13pm = e_13p * e_m
# e_23pm = e_23p * e_m
#
# # Base mbjects mf grade 5
# e_123pm = e_123p * e_m
