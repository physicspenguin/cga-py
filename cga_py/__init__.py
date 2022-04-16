from .base_objects import (
    e_1,
    e_2,
    e_3,
    e_i,
    e_o,
    e_12,
    e_13,
    e_1i,
    e_1o,
    e_23,
    e_2i,
    e_2o,
    e_3i,
    e_3o,
    e_io,
    e_123,
    e_12i,
    e_12o,
    e_13i,
    e_13o,
    e_1io,
    e_23i,
    e_23o,
    e_2io,
    e_3io,
    e_123i,
    e_123o,
    e_12io,
    e_13io,
    e_23io,
    e_123io,
    eps_1,
    eps_2,
    eps_3,
    q_i,
    q_j,
    q_k,
)
from .conditions import study_cond, study_var, null_quadric
from .four_quaternions import (
    arr_to_quat,
    quat_to_arr,
    rotor_to_quat,
    quat_to_rotor,
    vectorial,
    scalar,
)
from .geom_generators import (
    point,
    normalize_point,
    point_to_cartesian,
    sphere,
    normalize_sphere,
    sphere_to_cartesian,
    plane,
    normalize_plane,
    plane_to_cartesian,
)
from .multivector import cga_object
from .operators import act, com, anti_com, n_grade, r_norm, l_norm
from .permutations import *
from .random import (
    rand_rational,
    rand_point,
    rand_sphere,
    rand_plane,
    rand_plane,
    rand_rotor,
    rand_rot_poly,
    rand_zero,
)
from .kinematic_polynomials import poly_act
from .dorst_motions import iso_scale, transv
from .visualization import point_p_act
