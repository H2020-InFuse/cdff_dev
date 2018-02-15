import numpy as np
from cdff_dev import cdff_types
from nose.tools import assert_equal, assert_greater, assert_regexp_matches
from numpy.testing import assert_array_equal


def test_get_set_microseconds():
    t = cdff_types.Time()
    m = 1000023
    t.microseconds = m
    assert_equal(t.microseconds, m)


def test_time_str():
    t = cdff_types.Time()
    assert_regexp_matches(str(t), "{type: Time, microseconds: \d+, usec_per_sec: \d+}")


def test_vector2d_len():
    v = cdff_types.Vector2d()
    assert_equal(len(v), 2)


def test_vector2d_set_item():
    v = cdff_types.Vector2d()
    v[0] = 0.0
    v[1] = 1.0
    assert_array_equal(v, np.array([0.0, 1.0]))


def test_vector2d_str():
    v = cdff_types.Vector2d()
    v[0] = 0.0
    v[1] = 1.0
    assert_equal(str(v), "{type: Vector2d, data=[0.00, 1.00]}")


def test_vector2d_get_item():
    v = cdff_types.Vector2d()
    v[0] = 0.0
    v[1] = 1.0
    assert_equal(v[0], 0.0)
    assert_equal(v[1], 1.0)


def test_vector2d_assign():
    v1 = cdff_types.Vector2d()
    v1[0] = 0.0
    v1[1] = 1.0
    v2 = cdff_types.Vector2d()
    v2.assign(v1)
    assert_array_equal(v1, v2)


def test_vector2d_array():
    v = cdff_types.Vector2d()
    v[0] = 0.0
    v[1] = 1.0
    assert_array_equal(v.__array__(), np.array([0.0, 1.0]))


def test_vector2d_toarray():
    v = cdff_types.Vector2d()
    v[0] = 0.0
    v[1] = 1.0
    array = v.toarray()
    assert_array_equal(array, np.array([0.0, 1.0]))
    assert_equal(type(array), np.ndarray)


def test_vector2d_fromarray():
    v1 = np.array([0.0, 1.0])
    v2 = cdff_types.Vector2d()
    v2.fromarray(v1)
    assert_array_equal(v1, v2)


def test_vector3d_len():
    v = cdff_types.Vector3d()
    assert_equal(len(v), 3)


def test_vector3d_set_item():
    v = cdff_types.Vector3d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    assert_array_equal(v, np.array([0.0, 1.0, 2.0]))


def test_vector3d_str():
    v = cdff_types.Vector3d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    assert_equal(str(v), "{type: Vector3d, data=[0.00, 1.00, 2.00]}")


def test_vector3d_get_item():
    v = cdff_types.Vector3d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    assert_equal(v[0], 0.0)
    assert_equal(v[1], 1.0)
    assert_equal(v[2], 2.0)


def test_vector3d_assign():
    v1 = cdff_types.Vector3d()
    v1[0] = 0.0
    v1[1] = 1.0
    v1[2] = 2.0
    v2 = cdff_types.Vector3d()
    v2.assign(v1)
    assert_array_equal(v1, v2)


def test_vector3d_array():
    v = cdff_types.Vector3d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    assert_array_equal(v.__array__(), np.array([0.0, 1.0, 2.0]))


def test_vector3d_toarray():
    v = cdff_types.Vector3d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    array = v.toarray()
    assert_array_equal(array, np.array([0.0, 1.0, 2.0]))
    assert_equal(type(array), np.ndarray)


def test_vector3d_fromarray():
    v1 = np.array([0.0, 1.0, 2.0])
    v2 = cdff_types.Vector3d()
    v2.fromarray(v1)
    assert_array_equal(v1, v2)


def test_vector4d_len():
    v = cdff_types.Vector4d()
    assert_equal(len(v), 4)


def test_vector4d_set_item():
    v = cdff_types.Vector4d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    assert_array_equal(v, np.array([0.0, 1.0, 2.0, 3.0]))


def test_vector4d_str():
    v = cdff_types.Vector4d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    assert_equal(str(v), "{type: Vector4d, data=[0.00, 1.00, 2.00, 3.00]}")


def test_vector4d_get_item():
    v = cdff_types.Vector4d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    assert_equal(v[0], 0.0)
    assert_equal(v[1], 1.0)
    assert_equal(v[2], 2.0)


def test_vector4d_assign():
    v1 = cdff_types.Vector4d()
    v1[0] = 0.0
    v1[1] = 1.0
    v1[2] = 2.0
    v1[3] = 3.0
    v2 = cdff_types.Vector4d()
    v2.assign(v1)
    assert_array_equal(v1, v2)


def test_vector4d_array():
    v = cdff_types.Vector4d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    assert_array_equal(v.__array__(), np.array([0.0, 1.0, 2.0, 3.0]))


def test_vector4d_toarray():
    v = cdff_types.Vector4d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    array = v.toarray()
    assert_array_equal(array, np.array([0.0, 1.0, 2.0, 3.0]))
    assert_equal(type(array), np.ndarray)


def test_vector4d_fromarray():
    v1 = np.array([0.0, 1.0, 2.0, 3.0])
    v2 = cdff_types.Vector4d()
    v2.fromarray(v1)
    assert_array_equal(v1, v2)
