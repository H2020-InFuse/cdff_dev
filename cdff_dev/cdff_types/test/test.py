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


def test_vector6d_len():
    v = cdff_types.Vector6d()
    assert_equal(len(v), 6)


def test_vector6d_set_item():
    v = cdff_types.Vector6d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    assert_array_equal(v, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


def test_vector6d_str():
    v = cdff_types.Vector6d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    assert_equal(
        str(v), "{type: Vector6d, data=[0.00, 1.00, 2.00, 3.00, 4.00, 5.00]}")


def test_vector6d_get_item():
    v = cdff_types.Vector6d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    assert_equal(v[0], 0.0)
    assert_equal(v[1], 1.0)
    assert_equal(v[2], 2.0)
    assert_equal(v[3], 3.0)
    assert_equal(v[4], 4.0)
    assert_equal(v[5], 5.0)


def test_vector6d_assign():
    v1 = cdff_types.Vector6d()
    v1[0] = 0.0
    v1[1] = 1.0
    v1[2] = 2.0
    v1[3] = 3.0
    v1[4] = 4.0
    v1[5] = 5.0
    v2 = cdff_types.Vector6d()
    v2.assign(v1)
    assert_array_equal(v1, v2)


def test_vector6d_array():
    v = cdff_types.Vector6d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    assert_array_equal(v.__array__(), np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


def test_vector6d_toarray():
    v = cdff_types.Vector6d()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    array = v.toarray()
    assert_array_equal(array, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert_equal(type(array), np.ndarray)


def test_vector6d_fromarray():
    v1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    v2 = cdff_types.Vector6d()
    v2.fromarray(v1)
    assert_array_equal(v1, v2)


def test_vectorXd_len():
    v = cdff_types.VectorXd(6)
    assert_equal(len(v), 6)


def test_vectorXd_set_item():
    v = cdff_types.VectorXd(6)
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    assert_array_equal(v, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


def test_vectorXd_str():
    v = cdff_types.VectorXd(6)
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    assert_equal(
        str(v), "{type: VectorXd, data=[0.00, 1.00, 2.00, 3.00, 4.00, 5.00]}")


def test_vectorXd_get_item():
    v = cdff_types.VectorXd(6)
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    assert_equal(v[0], 0.0)
    assert_equal(v[1], 1.0)
    assert_equal(v[2], 2.0)
    assert_equal(v[3], 3.0)
    assert_equal(v[4], 4.0)
    assert_equal(v[5], 5.0)


def test_vectorXd_assign():
    v1 = cdff_types.VectorXd(6)
    v1[0] = 0.0
    v1[1] = 1.0
    v1[2] = 2.0
    v1[3] = 3.0
    v1[4] = 4.0
    v1[5] = 5.0
    v2 = cdff_types.VectorXd(6)
    v2.assign(v1)
    assert_array_equal(v1, v2)


def test_vectorXd_array():
    v = cdff_types.VectorXd(6)
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    assert_array_equal(v.__array__(), np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


def test_vectorXd_toarray():
    v = cdff_types.VectorXd(6)
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    v[4] = 4.0
    v[5] = 5.0
    array = v.toarray()
    assert_array_equal(array, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert_equal(type(array), np.ndarray)


def test_vectorXd_fromarray():
    v1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    v2 = cdff_types.VectorXd(6)
    v2.fromarray(v1)
    assert_array_equal(v1, v2)


def test_matrix2d_len():
    v = cdff_types.Matrix2d()
    assert_equal(len(v), 2)


def test_matrix2d_set_item():
    v = cdff_types.Matrix2d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[0, 1] = 2.0
    v[1, 1] = 3.0
    assert_array_equal(v, np.array([[0.0, 2.0], [1.0, 3.0]]))


def test_matrix2d_str():
    v = cdff_types.Matrix2d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[0, 1] = 2.0
    v[1, 1] = 3.0
    assert_equal(
        str(v), "{type: Matrix2d, data=[?]}")


def test_matrix2d_get_item():
    v = cdff_types.Matrix2d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[0, 1] = 2.0
    v[1, 1] = 3.0
    assert_equal(v[0, 0], 0.0)
    assert_equal(v[1, 0], 1.0)
    assert_equal(v[0, 1], 2.0)
    assert_equal(v[1, 1], 3.0)


def test_matrix2d_assign():
    v1 = cdff_types.Matrix2d()
    v1[0, 0] = 0.0
    v1[1, 0] = 1.0
    v1[0, 1] = 2.0
    v1[1, 1] = 3.0
    v2 = cdff_types.Matrix2d()
    v2.assign(v1)
    assert_array_equal(v1, v2)


def test_matrix2d_array():
    v = cdff_types.Matrix2d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[0, 1] = 2.0
    v[1, 1] = 3.0
    assert_array_equal(v.__array__(), np.array([[0.0, 2.0], [1.0, 3.0]]))


def test_matrix2d_toarray():
    v = cdff_types.Matrix2d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[0, 1] = 2.0
    v[1, 1] = 3.0
    array = v.toarray()
    assert_array_equal(array, np.array([[0.0, 2.0], [1.0, 3.0]]))
    assert_equal(type(array), np.ndarray)


def test_matrix2d_fromarray():
    v1 = np.array([[0.0, 2.0], [1.0, 3.0]])
    v2 = cdff_types.Matrix2d()
    v2.fromarray(v1)
    assert_array_equal(v1, v2)


def test_matrix3d_len():
    v = cdff_types.Matrix3d()
    assert_equal(len(v), 3)


def test_matrix3d_set_item():
    v = cdff_types.Matrix3d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[2, 0] = 2.0
    v[0, 1] = 3.0
    v[1, 1] = 4.0
    v[2, 1] = 5.0
    v[0, 2] = 6.0
    v[1, 2] = 7.0
    v[2, 2] = 8.0
    assert_array_equal(
        v, np.array([[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]]))


def test_matrix3d_str():
    v = cdff_types.Matrix3d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[2, 0] = 2.0
    v[0, 1] = 3.0
    v[1, 1] = 4.0
    v[2, 1] = 5.0
    v[0, 2] = 6.0
    v[1, 2] = 7.0
    v[2, 2] = 8.0
    assert_equal(
        str(v), "{type: Matrix3d, data=[?]}")


def test_matrix3d_get_item():
    v = cdff_types.Matrix3d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[2, 0] = 2.0
    v[0, 1] = 3.0
    v[1, 1] = 4.0
    v[2, 1] = 5.0
    v[0, 2] = 6.0
    v[1, 2] = 7.0
    v[2, 2] = 8.0
    assert_equal(v[0, 0], 0.0)
    assert_equal(v[1, 0], 1.0)
    assert_equal(v[2, 0], 2.0)
    assert_equal(v[0, 1], 3.0)
    assert_equal(v[1, 1], 4.0)
    assert_equal(v[2, 1], 5.0)
    assert_equal(v[0, 2], 6.0)
    assert_equal(v[1, 2], 7.0)
    assert_equal(v[2, 2], 8.0)


def test_matrix3d_assign():
    v1 = cdff_types.Matrix3d()
    v1[0, 0] = 0.0
    v1[1, 0] = 1.0
    v1[2, 0] = 2.0
    v1[0, 1] = 3.0
    v1[1, 1] = 4.0
    v1[2, 1] = 5.0
    v1[0, 2] = 6.0
    v1[1, 2] = 7.0
    v1[2, 2] = 8.0
    v2 = cdff_types.Matrix3d()
    v2.assign(v1)
    assert_array_equal(v1, v2)


def test_matrix3d_array():
    v = cdff_types.Matrix3d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[2, 0] = 2.0
    v[0, 1] = 3.0
    v[1, 1] = 4.0
    v[2, 1] = 5.0
    v[0, 2] = 6.0
    v[1, 2] = 7.0
    v[2, 2] = 8.0
    assert_array_equal(
        v.__array__(),
        np.array([[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]]))


def test_matrix3d_toarray():
    v = cdff_types.Matrix3d()
    v[0, 0] = 0.0
    v[1, 0] = 1.0
    v[2, 0] = 2.0
    v[0, 1] = 3.0
    v[1, 1] = 4.0
    v[2, 1] = 5.0
    v[0, 2] = 6.0
    v[1, 2] = 7.0
    v[2, 2] = 8.0
    array = v.toarray()
    assert_array_equal(
        array, np.array([[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]]))
    assert_equal(type(array), np.ndarray)


def test_matrix3d_fromarray():
    v1 = np.array([[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]])
    v2 = cdff_types.Matrix3d()
    v2.fromarray(v1)
    assert_array_equal(v1, v2)


def test_quaterniond_len():
    v = cdff_types.Quaterniond()
    assert_equal(len(v), 4)


def test_quaterniond_set_item():
    v = cdff_types.Quaterniond()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    assert_array_equal(v, np.array([0.0, 1.0, 2.0, 3.0]))


def test_quaterniond_str():
    v = cdff_types.Quaterniond()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    assert_equal(str(v), "{type: Quaterniond, data=[0.00, 1.00, 2.00, 3.00]}")


def test_quaterniond_get_item():
    v = cdff_types.Quaterniond()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    assert_equal(v[0], 0.0)
    assert_equal(v[1], 1.0)
    assert_equal(v[2], 2.0)


def test_quaterniond_assign():
    v1 = cdff_types.Quaterniond()
    v1[0] = 0.0
    v1[1] = 1.0
    v1[2] = 2.0
    v1[3] = 3.0
    v2 = cdff_types.Quaterniond()
    v2.assign(v1)
    assert_array_equal(v1, v2)


def test_quaterniond_array():
    v = cdff_types.Quaterniond()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    assert_array_equal(v.__array__(), np.array([0.0, 1.0, 2.0, 3.0]))


def test_quaterniond_toarray():
    v = cdff_types.Quaterniond()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    array = v.toarray()
    assert_array_equal(array, np.array([0.0, 1.0, 2.0, 3.0]))
    assert_equal(type(array), np.ndarray)


def test_quaterniond_fromarray():
    v1 = np.array([0.0, 1.0, 2.0, 3.0])
    v2 = cdff_types.Quaterniond()
    v2.fromarray(v1)
    assert_array_equal(v1, v2)