from .base_objects import (
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
)

# Permutations of order 2 elements
e_12 = e_12
e_13 = e_13
e_1i = e_1i
e_1o = e_1o
e_23 = e_23
e_2i = e_2i
e_2o = e_2o
e_3i = e_3i
e_3o = e_3o
e_io = e_io
e_21 = -e_12
e_31 = -e_13
e_i1 = -e_1i
e_o1 = -e_1o
e_32 = -e_23
e_i2 = -e_2i
e_o2 = -e_2o
e_i3 = -e_3i
e_o3 = -e_3o
e_oi = -e_io


# Permutations of order 3 elements
e_123 = e_123
e_12i = e_12i
e_12o = e_12o
e_13i = e_13i
e_13o = e_13o
e_1io = e_1io
e_23i = e_23i
e_23o = e_23o
e_2io = e_2io
e_3io = e_3io
e_132 = -e_123
e_1i2 = -e_12i
e_1o2 = -e_12o
e_1i3 = -e_13i
e_1o3 = -e_13o
e_1oi = -e_1io
e_2i3 = -e_23i
e_2o3 = -e_23o
e_2oi = -e_2io
e_3oi = -e_3io
e_213 = -e_123
e_21i = -e_12i
e_21o = -e_12o
e_31i = -e_13i
e_31o = -e_13o
e_i1o = -e_1io
e_32i = -e_23i
e_32o = -e_23o
e_i2o = -e_2io
e_i3o = -e_3io
e_231 = e_123
e_2i1 = e_12i
e_2o1 = e_12o
e_3i1 = e_13i
e_3o1 = e_13o
e_io1 = e_1io
e_3i2 = e_23i
e_3o2 = e_23o
e_io2 = e_2io
e_io3 = e_3io
e_312 = e_123
e_i12 = e_12i
e_o12 = e_12o
e_i13 = e_13i
e_o13 = e_13o
e_o1i = e_1io
e_i23 = e_23i
e_o23 = e_23o
e_o2i = e_2io
e_o3i = e_3io
e_321 = -e_123
e_i21 = -e_12i
e_o21 = -e_12o
e_i31 = -e_13i
e_o31 = -e_13o
e_oi1 = -e_1io
e_i32 = -e_23i
e_o32 = -e_23o
e_oi2 = -e_2io
e_oi3 = -e_3io


# Permutations of order 4 elements
e_123i = e_123i
e_123o = e_123o
e_12io = e_12io
e_13io = e_13io
e_23io = e_23io
e_12i3 = -e_123i
e_12o3 = -e_123o
e_12oi = -e_12io
e_13oi = -e_13io
e_23oi = -e_23io
e_132i = -e_123i
e_132o = -e_123o
e_1i2o = -e_12io
e_1i3o = -e_13io
e_2i3o = -e_23io
e_13i2 = e_123i
e_13o2 = e_123o
e_1io2 = e_12io
e_1io3 = e_13io
e_2io3 = e_23io
e_1i23 = e_123i
e_1o23 = e_123o
e_1o2i = e_12io
e_1o3i = e_13io
e_2o3i = e_23io
e_1i32 = -e_123i
e_1o32 = -e_123o
e_1oi2 = -e_12io
e_1oi3 = -e_13io
e_2oi3 = -e_23io
e_213i = -e_123i
e_213o = -e_123o
e_21io = -e_12io
e_31io = -e_13io
e_32io = -e_23io
e_21i3 = e_123i
e_21o3 = e_123o
e_21oi = e_12io
e_31oi = e_13io
e_32oi = e_23io
e_231i = e_123i
e_231o = e_123o
e_2i1o = e_12io
e_3i1o = e_13io
e_3i2o = e_23io
e_23i1 = -e_123i
e_23o1 = -e_123o
e_2io1 = -e_12io
e_3io1 = -e_13io
e_3io2 = -e_23io
e_2i13 = -e_123i
e_2o13 = -e_123o
e_2o1i = -e_12io
e_3o1i = -e_13io
e_3o2i = -e_23io
e_2i31 = e_123i
e_2o31 = e_123o
e_2oi1 = e_12io
e_3oi1 = e_13io
e_3oi2 = e_23io
e_312i = e_123i
e_312o = e_123o
e_i12o = e_12io
e_i13o = e_13io
e_i23o = e_23io
e_31i2 = -e_123i
e_31o2 = -e_123o
e_i1o2 = -e_12io
e_i1o3 = -e_13io
e_i2o3 = -e_23io
e_321i = -e_123i
e_321o = -e_123o
e_i21o = -e_12io
e_i31o = -e_13io
e_i32o = -e_23io
e_32i1 = e_123i
e_32o1 = e_123o
e_i2o1 = e_12io
e_i3o1 = e_13io
e_i3o2 = e_23io
e_3i12 = e_123i
e_3o12 = e_123o
e_io12 = e_12io
e_io13 = e_13io
e_io23 = e_23io
e_3i21 = -e_123i
e_3o21 = -e_123o
e_io21 = -e_12io
e_io31 = -e_13io
e_io32 = -e_23io
e_i123 = -e_123i
e_o123 = -e_123o
e_o12i = -e_12io
e_o13i = -e_13io
e_o23i = -e_23io
e_i132 = e_123i
e_o132 = e_123o
e_o1i2 = e_12io
e_o1i3 = e_13io
e_o2i3 = e_23io
e_i213 = e_123i
e_o213 = e_123o
e_o21i = e_12io
e_o31i = e_13io
e_o32i = e_23io
e_i231 = -e_123i
e_o231 = -e_123o
e_o2i1 = -e_12io
e_o3i1 = -e_13io
e_o3i2 = -e_23io
e_i312 = -e_123i
e_o312 = -e_123o
e_oi12 = -e_12io
e_oi13 = -e_13io
e_oi23 = -e_23io
e_i321 = e_123i
e_o321 = e_123o
e_oi21 = e_12io
e_oi31 = e_13io
e_oi32 = e_23io


# Permutations of order 5 elements
e_123io = e_123io
e_123oi = -e_123io
e_12i3o = -e_123io
e_12io3 = e_123io
e_12o3i = e_123io
e_12oi3 = -e_123io
e_132io = -e_123io
e_132oi = e_123io
e_13i2o = e_123io
e_13io2 = -e_123io
e_13o2i = -e_123io
e_13oi2 = e_123io
e_1i23o = e_123io
e_1i2o3 = -e_123io
e_1i32o = -e_123io
e_1i3o2 = e_123io
e_1io23 = e_123io
e_1io32 = -e_123io
e_1o23i = -e_123io
e_1o2i3 = e_123io
e_1o32i = e_123io
e_1o3i2 = -e_123io
e_1oi23 = -e_123io
e_1oi32 = e_123io
e_213io = -e_123io
e_213oi = e_123io
e_21i3o = e_123io
e_21io3 = -e_123io
e_21o3i = -e_123io
e_21oi3 = e_123io
e_231io = e_123io
e_231oi = -e_123io
e_23i1o = -e_123io
e_23io1 = e_123io
e_23o1i = e_123io
e_23oi1 = -e_123io
e_2i13o = -e_123io
e_2i1o3 = e_123io
e_2i31o = e_123io
e_2i3o1 = -e_123io
e_2io13 = -e_123io
e_2io31 = e_123io
e_2o13i = e_123io
e_2o1i3 = -e_123io
e_2o31i = -e_123io
e_2o3i1 = e_123io
e_2oi13 = e_123io
e_2oi31 = -e_123io
e_312io = e_123io
e_312oi = -e_123io
e_31i2o = -e_123io
e_31io2 = e_123io
e_31o2i = e_123io
e_31oi2 = -e_123io
e_321io = -e_123io
e_321oi = e_123io
e_32i1o = e_123io
e_32io1 = -e_123io
e_32o1i = -e_123io
e_32oi1 = e_123io
e_3i12o = e_123io
e_3i1o2 = -e_123io
e_3i21o = -e_123io
e_3i2o1 = e_123io
e_3io12 = e_123io
e_3io21 = -e_123io
e_3o12i = -e_123io
e_3o1i2 = e_123io
e_3o21i = e_123io
e_3o2i1 = -e_123io
e_3oi12 = -e_123io
e_3oi21 = e_123io
e_i123o = -e_123io
e_i12o3 = e_123io
e_i132o = e_123io
e_i13o2 = -e_123io
e_i1o23 = -e_123io
e_i1o32 = e_123io
e_i213o = e_123io
e_i21o3 = -e_123io
e_i231o = -e_123io
e_i23o1 = e_123io
e_i2o13 = e_123io
e_i2o31 = -e_123io
e_i312o = -e_123io
e_i31o2 = e_123io
e_i321o = e_123io
e_i32o1 = -e_123io
e_i3o12 = -e_123io
e_i3o21 = e_123io
e_io123 = e_123io
e_io132 = -e_123io
e_io213 = -e_123io
e_io231 = e_123io
e_io312 = e_123io
e_io321 = -e_123io
e_o123i = e_123io
e_o12i3 = -e_123io
e_o132i = -e_123io
e_o13i2 = e_123io
e_o1i23 = e_123io
e_o1i32 = -e_123io
e_o213i = -e_123io
e_o21i3 = e_123io
e_o231i = e_123io
e_o23i1 = -e_123io
e_o2i13 = -e_123io
e_o2i31 = e_123io
e_o312i = e_123io
e_o31i2 = -e_123io
e_o321i = -e_123io
e_o32i1 = e_123io
e_o3i12 = e_123io
e_o3i21 = -e_123io
e_oi123 = -e_123io
e_oi132 = e_123io
e_oi213 = e_123io
e_oi231 = -e_123io
e_oi312 = -e_123io
e_oi321 = e_123io
