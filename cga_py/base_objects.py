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
