import numpy as np
import pickle
import yaml
from cdff_dev.extensions.pcl import helpers
import cdff_types
from nose.tools import (assert_equal, assert_regexp_matches, assert_true,
                        assert_false)
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_get_set_microseconds():
    t = cdff_types.Time()
    m = 1000023
    t.microseconds = m
    assert_equal(t.microseconds, m)


def test_set_float():
    t = cdff_types.Time()
    m = 1000023.0
    t.microseconds = m
    assert_equal(t.microseconds, m)


def test_time_str():
    t = cdff_types.Time()
    assert_regexp_matches(str(t), "{type: Time, microseconds: \d+}")


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
    assert_equal(str(v), "{type: Vector2d, data: [0, 1]}")


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
    assert_equal(str(v), "{type: Vector3d, data: [0, 1, 2]}")


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
    assert_equal(str(v), "{type: Vector4d, data: [0, 1, 2, 3]}")


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
        str(v), "{type: Vector6d, data: [0, 1, 2, 3, 4, 5]}")


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
        str(v), "{type: VectorXd, data: [0, 1, 2, 3, 4, 5]}")


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
        str(v), "{type: Matrix2d, data: [[0, 2], [1, 3]]}")


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
        str(v), "{type: Matrix3d, data: [[0, 3, 6], [1, 4, 7], [2, 5, 8]]}")


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


def test_matrix6d_len():
    v = cdff_types.Matrix6d()
    assert_equal(len(v), 6)


def test_matrix6d_set_item():
    v = cdff_types.Matrix6d()
    for i in range(6):
        for j in range(6):
            v[i, j] = i * 6.0 + j
    assert_array_equal(
        v, np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                     [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                     [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                     [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                     [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
                     [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]]))


def test_matrix6d_str():
    v = cdff_types.Matrix6d()
    for i in range(6):
        for j in range(6):
            v[i, j] = i * 6.0 + j
    assert_equal(
        str(v),
        "{type: Matrix6d, data: "
        "[[0, 1, 2, 3, 4, 5], "
        "[6, 7, 8, 9, 10, 11], "
        "[12, 13, 14, 15, 16, 17], "
        "[18, 19, 20, 21, 22, 23], "
        "[24, 25, 26, 27, 28, 29], "
        "[30, 31, 32, 33, 34, 35]]}")


def test_matrix6d_get_item():
    v = cdff_types.Matrix6d()
    for i in range(6):
        for j in range(6):
            v[i, j] = i * 6.0 + j
    assert_equal(v[0, 0], 0.0)
    assert_equal(v[1, 0], 6.0)
    assert_equal(v[2, 0], 12.0)
    assert_equal(v[0, 1], 1.0)
    assert_equal(v[1, 1], 7.0)
    assert_equal(v[2, 1], 13.0)
    assert_equal(v[0, 2], 2.0)
    assert_equal(v[1, 2], 8.0)
    assert_equal(v[2, 2], 14.0)


def test_matrix6d_array():
    v = cdff_types.Matrix6d()
    for i in range(6):
        for j in range(6):
            v[i, j] = i * 6.0 + j
    assert_array_equal(
        v.__array__(),
        np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                  [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                  [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                  [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
                  [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]]))


def test_matrix6d_toarray():
    v = cdff_types.Matrix6d()
    for i in range(6):
        for j in range(6):
            v[i, j] = i * 6.0 + j
    array = v.toarray()
    assert_array_equal(
        array,
        np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                  [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                  [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                  [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
                  [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]]))
    assert_equal(type(array), np.ndarray)


def test_matrix6d_fromarray():
    v1 = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                  [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                  [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                  [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
                  [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]])
    v2 = cdff_types.Matrix6d()
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
    assert_equal(
        str(v), "{type: Quaterniond, data: {x: 0, y: 1, z: 2, w: 3}}")


def test_quaterniond_get_item():
    v = cdff_types.Quaterniond()
    v[0] = 0.0
    v[1] = 1.0
    v[2] = 2.0
    v[3] = 3.0
    assert_equal(v[0], 0.0)
    assert_equal(v[1], 1.0)
    assert_equal(v[2], 2.0)
    assert_equal(v[3], 3.0)


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


def test_quaternion_conjugate_product():
    q = cdff_types.Quaterniond()
    q.fromarray(np.array([0.13155995, 0.32178034, 0.73673994, 0.57996866]))
    assert_array_almost_equal(
        (q * q.conjugate()).toarray(), np.array([0.0, 0.0, 0.0, 1.0]))


def test_transform_pose():
    p = cdff_types.Pose()

    p.pos.fromarray(np.array([1.0, 2.0, 3.0]))
    assert_array_equal(p.pos, np.array([1.0, 2.0, 3.0]))

    p.orient.fromarray(np.array([1.0, 2.0, 3.0, 4.0]))
    assert_array_equal(p.orient, np.array([1.0, 2.0, 3.0, 4.0]))


def test_transform_with_covariance():
    t = cdff_types.TransformWithCovariance()

    t.metadata.msg_version = 5
    assert_equal(t.metadata.msg_version, 5)

    t.metadata.producer_id = "producer"
    assert_equal(t.metadata.producer_id, "producer")

    t.metadata.parent_frame_id = "parent"
    assert_equal(t.metadata.parent_frame_id, "parent")

    t.metadata.parent_time.microseconds = 5
    assert_equal(t.metadata.parent_time.microseconds, 5)

    t.metadata.child_frame_id = "child"
    assert_equal(t.metadata.child_frame_id, "child")

    t.metadata.child_time.microseconds = 10
    assert_equal(t.metadata.child_time.microseconds, 10)

    t.data.translation.fromarray(np.array([1.0, 2.0, 3.0]))
    assert_array_equal(t.data.translation, np.array([1.0, 2.0, 3.0]))

    t.data.orientation.fromarray(np.array([1.0, 2.0, 3.0, 4.0]))
    assert_array_equal(t.data.orientation, np.array([1.0, 2.0, 3.0, 4.0]))

    cov = np.array(
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
         [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
         [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
         [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
         [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
         [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]])
    t.data.cov.fromarray(cov)
    assert_array_equal(t.data.cov, cov)


def test_create_pointcloud():
    pcl = cdff_types.Pointcloud()

    assert_equal(pcl.metadata.msg_version, 1)
    pcl.metadata.msg_version = 5
    assert_equal(pcl.metadata.msg_version, 5)

    pcl.metadata.sensor_id = "sensor"
    assert_equal(pcl.metadata.sensor_id, "sensor")

    pcl.metadata.frame_id = "frame"
    assert_equal(pcl.metadata.frame_id, "frame")

    pcl.metadata.time_stamp.microseconds = 10
    assert_equal(pcl.metadata.time_stamp.microseconds, 10)

    pcl.metadata.height = 20
    assert_equal(pcl.metadata.height, 20)

    pcl.metadata.width = 30
    assert_equal(pcl.metadata.width, 30)

    pcl.metadata.is_registered = True
    assert_true(pcl.metadata.is_registered)

    pcl.metadata.is_ordered = False
    assert_false(pcl.metadata.is_ordered)

    pcl.metadata.has_fixed_transform = True
    assert_true(pcl.metadata.has_fixed_transform)

    pcl.metadata.pose_robot_frame_sensor_frame.data.translation.fromarray(
        np.array([1.0, 2.0, 3.0]))
    assert_array_equal(
        pcl.metadata.pose_robot_frame_sensor_frame.data.translation.toarray(),
        np.array([1.0, 2.0, 3.0]))

    pcl.metadata.pose_fixed_frame_robot_frame.data.translation.fromarray(
        np.array([1.0, 2.0, 3.0]))
    assert_array_equal(
        pcl.metadata.pose_fixed_frame_robot_frame.data.translation.toarray(),
        np.array([1.0, 2.0, 3.0]))

    pcl.data.points.resize(100)
    pcl.data.points[0, 0] = 1.0
    pcl.data.points[0, 1] = 2.0
    pcl.data.points[0, 2] = 3.0
    assert_equal(pcl.data.points.size(), 100)
    assert_equal(pcl.data.points[0, 0], 1.0)
    assert_equal(pcl.data.points[0, 1], 2.0)
    assert_equal(pcl.data.points[0, 2], 3.0)

    pcl.data.colors.resize(100)
    pcl.data.colors[0, 0] = 255.0
    pcl.data.colors[0, 1] = 255.0
    pcl.data.colors[0, 2] = 255.0
    assert_equal(pcl.data.colors.size(), 100)
    assert_equal(pcl.data.colors[0, 0], 255.0)
    assert_equal(pcl.data.colors[0, 1], 255.0)
    assert_equal(pcl.data.colors[0, 2], 255.0)


def test_filter_pointcloud():
    pcl = cdff_types.Pointcloud()
    pcl.data.points.resize(5)
    pcl.data.points[0, 0] = 0.0
    pcl.data.points[0, 1] = 0.0
    pcl.data.points[0, 2] = 0.0
    for i in range(1, 5):
        pcl.data.points[i, 0] = np.inf
        pcl.data.points[i, 1] = np.inf
        pcl.data.points[i, 2] = np.inf
    assert_equal(pcl.data.points.size(), 5)
    pcl_filtered = pcl.filtered()
    assert_equal(pcl_filtered.data.points.size(), 1)


def test_load_ply():
    pc = helpers.load_ply_file(
        "test/test_data/pointclouds/cube.ply")
    assert_equal(pc.data.points.size(), 4)
    pc = helpers.load_ply_file(
        "test/test_data/pointclouds/dense_original.ply")
    assert_equal(pc.data.points.size(), 5824)


def test_create_laserscan():
    ls = cdff_types.LaserScan()

    ls.ref_time.microseconds = 0
    assert_equal(ls.ref_time.microseconds, 0)

    ls.start_angle = 1.0
    assert_equal(ls.start_angle, 1.0)

    ls.angular_resolution = 1.0
    assert_equal(ls.angular_resolution, 1.0)

    ls.speed = 1.0
    assert_equal(ls.speed, 1.0)

    ls.max_range = 1.0
    assert_equal(ls.max_range, 1.0)

    ls.min_range = 1.0
    assert_equal(ls.min_range, 1.0)

    ls.remission.resize(60)
    ls.remission[0] = 1.0
    ls.remission[1] = 2.0
    ls.remission[2] = 3.0
    assert_equal(ls.remission.size(), 60)
    assert_array_equal(np.asarray([ls.remission[i] for i in range(3)]),
       (1.0, 2.0, 3.0))

    ls.ranges.resize(60)
    ls.ranges[0] = 1
    ls.ranges[1] = 2
    ls.ranges[2] = 3
    assert_equal(ls.ranges.size(), 60)
    assert_array_equal(np.asarray([ls.ranges[i] for i in range(3)]),
       (1, 2, 3))


def test_rigid_body_state_get_set_time():
    rbs = cdff_types.RigidBodyState()
    assert_equal(rbs.timestamp.microseconds, 0)
    rbs.timestamp.microseconds = 500
    assert_equal(rbs.timestamp.microseconds, 500)
    time = cdff_types.Time()
    time.microseconds = 1000
    rbs.timestamp = time
    assert_equal(rbs.timestamp.microseconds, 1000)


def test_rigid_body_state_get_set_source_frame():
    rbs = cdff_types.RigidBodyState()
    assert_equal(rbs.source_frame, "")
    rbs.source_frame = "source_frame"
    assert_equal(rbs.source_frame, "source_frame")


def test_rigid_body_state_get_set_target_frame():
    rbs = cdff_types.RigidBodyState()
    assert_equal(rbs.target_frame, "")
    rbs.target_frame = "target_frame"
    assert_equal(rbs.target_frame, "target_frame")


def test_rigid_body_state_str():
    rbs = cdff_types.RigidBodyState()
    rbs.source_frame = "source"
    rbs.target_frame = "target"
    assert_equal.__self__.maxDiff = None
    assert_equal(
        str(rbs),
        "{type: RigidBodyState, timestamp={type: Time, microseconds: 0}, "
        "sourceFrame=source, targetFrame=target, pos={type: Vector3d, "
        "data: [0, 0, 0]}, orient={type: Quaterniond, "
        "data: {x: 0, y: 0, z: 0, w: 0}}, ...}")


def test_rigid_body_state_get_set_position():
    rbs = cdff_types.RigidBodyState()
    rbs.pos[0] = 1.0
    rbs.pos[1] = 2.0
    rbs.pos[2] = 3.0
    assert_array_almost_equal(rbs.pos.toarray(), np.array([1, 2, 3]))


def tst_rigid_body_state_get_set_cov_position():
    rbs = cdff_types.RigidBodyState()
    assert_array_almost_equal(
        rbs.cov_position.toarray(), np.ones((3, 3)) * np.nan)
    rbs.cov_position.fromarray(np.eye(3))
    assert_array_almost_equal(rbs.cov_position.toarray(), np.eye(3))


def test_rigid_body_state_get_set_orientation():
    rbs = cdff_types.RigidBodyState()
    rbs.orient.fromarray(np.array([1.0, 2.0, 3.0, 4.0]))
    assert_array_almost_equal(
        rbs.orient.toarray(), np.array([1.0, 2.0, 3.0, 4.0]))


def test_rigid_body_state_get_set_cov_orientation():
    rbs = cdff_types.RigidBodyState()
    rbs.cov_orientation.fromarray(np.eye(3))
    assert_array_almost_equal(rbs.cov_orientation.toarray(), np.eye(3))


def test_rigid_body_state_get_set_velocity():
    rbs = cdff_types.RigidBodyState()
    rbs.velocity[0] = 1.0
    rbs.velocity[1] = 2.0
    rbs.velocity[2] = 3.0
    assert_array_almost_equal(rbs.velocity.toarray(), np.array([1, 2, 3]))


def test_rigid_body_state_get_set_cov_velocity():
    rbs = cdff_types.RigidBodyState()
    rbs.cov_velocity.fromarray(np.eye(3))
    assert_array_almost_equal(rbs.cov_velocity.toarray(), np.eye(3))


def test_rigid_body_state_get_set_angular_velocity():
    rbs = cdff_types.RigidBodyState()
    rbs.angular_velocity[0] = 1.0
    rbs.angular_velocity[1] = 2.0
    rbs.angular_velocity[2] = 3.0
    assert_array_almost_equal(rbs.angular_velocity.toarray(), np.array([1, 2, 3]))


def test_rigid_body_state_get_set_cov_angular_velocity():
    rbs = cdff_types.RigidBodyState()
    rbs.cov_angular_velocity.fromarray(np.eye(3))
    assert_array_almost_equal(rbs.cov_angular_velocity.toarray(), np.eye(3))


def test_joint_state_get_set_position():
    js = cdff_types.JointState()
    js.position = 5.0
    assert_equal(js.position, 5.0)
    assert_equal(
        str(js), "{type: JointState, position: 5, speed: 0, "
        "effort: 0, raw: 0, acceleration: 0}")


def test_joint_state_get_set_speed():
    js = cdff_types.JointState()
    js.speed = 5.0
    assert_equal(js.speed, 5.0)
    assert_equal(
        str(js), "{type: JointState, position: 0, speed: 5, "
        "effort: 0, raw: 0, acceleration: 0}")


def test_joint_state_get_set_effort():
    js = cdff_types.JointState()
    js.effort = 5.0
    assert_equal(js.effort, 5.0)
    assert_equal(
        str(js), "{type: JointState, position: 0, speed: 0, "
        "effort: 5, raw: 0, acceleration: 0}")


def test_joint_state_get_set_raw():
    js = cdff_types.JointState()
    js.raw = 5.0
    assert_equal(js.raw, 5.0)
    assert_equal(
        str(js), "{type: JointState, position: 0, speed: 0, "
        "effort: 0, raw: 5, acceleration: 0}")


def test_joint_state_get_set_acceleration():
    js = cdff_types.JointState()
    js.acceleration = 5.0
    assert_equal(js.acceleration, 5.0)
    assert_equal(
        str(js), "{type: JointState, position: 0, speed: 0, "
        "effort: 0, raw: 0, acceleration: 5}")


def test_joints_get_set_time():
    joints = cdff_types.Joints()
    assert_equal(joints.timestamp.microseconds, 0)
    joints.timestamp.microseconds = 500
    assert_equal(joints.timestamp.microseconds, 500)
    time = cdff_types.Time()
    time.microseconds = 1000
    joints.timestamp = time
    assert_equal(joints.timestamp.microseconds, 1000)


def test_joints_get_set_names():
    joints = cdff_types.Joints()
    joints.names.resize(3)
    joints.names[0] = "joint0"
    joints.names[1] = "joint1"
    joints.names[2] = "joint2"
    assert_equal(joints.names[0], "joint0")
    assert_equal(joints.names[1], "joint1")
    assert_equal(joints.names[2], "joint2")


def test_joints_get_set_elements():
    joints = cdff_types.Joints()
    joints.elements.resize(2)
    js1 = cdff_types.JointState()
    js1.position = 1.0
    joints.elements[0] = js1
    js2 = cdff_types.JointState()
    js2.position = 2.0
    joints.elements[1] = js2
    assert_equal(joints.elements[0].position, 1.0)
    assert_equal(joints.elements[1].position, 2.0)


def test_imu_sensors():
    imu = cdff_types.IMUSensors()

    imu.timestamp = cdff_types.Time()
    imu.timestamp.microseconds = 10
    assert_equal(imu.timestamp.microseconds, 10)

    imu.acc[0] = 1.0
    assert_equal(imu.acc[0], 1.0)

    imu.gyro[0] = 2.0
    assert_equal(imu.gyro[0], 2.0)

    imu.mag[0] = 3.0
    assert_equal(imu.mag[0], 3.0)


def test_depth_map():
    depth_map = cdff_types.DepthMap()

    depth_map.ref_time = cdff_types.Time()
    depth_map.ref_time.microseconds = 10
    assert_equal(depth_map.ref_time.microseconds, 10)

    depth_map.distances[0] = 10.0
    assert_equal(depth_map.distances[0], 10.0)

    depth_map.remissions[0] = 10.0
    assert_equal(depth_map.remissions[0], 10.0)

    depth_map.vertical_interval[0] = 10.0
    assert_equal(depth_map.vertical_interval[0], 10.0)

    depth_map.timestamps[0].microseconds = 10
    assert_equal(depth_map.timestamps[0].microseconds, 10)

    depth_map.horizontal_interval[0] = 10.0
    assert_equal(depth_map.horizontal_interval[0], 10.0)

    depth_map.vertical_projection = "polar"
    assert_equal(depth_map.vertical_projection , "polar")

    depth_map.horizontal_projection = "planar"
    assert_equal(depth_map.horizontal_projection, "planar")

    depth_map.vertical_size = 10
    assert_equal(depth_map.vertical_size, 10)

    depth_map.horizontal_size = 10
    assert_equal(depth_map.horizontal_size, 10)


def test_frame():
    frame = cdff_types.Frame()

    assert_equal(frame.msg_version, 1)
    frame.msg_version = 41
    assert_equal(frame.msg_version, 41)

    frame.data.channels = 1
    assert_equal(frame.data.channels, 1)

    frame.data.cols = 2
    assert_equal(frame.data.cols, 2)

    frame.data.rows = 3
    assert_equal(frame.data.rows, 3)

    frame.data.depth = "depth_32F"
    assert_equal(frame.data.depth, "depth_32F")

    assert_equal(frame.data.msg_version, 1)
    frame.data.msg_version = 4
    assert_equal(frame.data.msg_version, 4)

    frame.data.row_size = 5
    assert_equal(frame.data.row_size, 5)

    frame.data.data[0] = 2
    frame.data.data[1] = 3
    frame.data.data[2] = 4
    assert_equal(frame.data.data[0], 2)
    assert_equal(frame.data.data[1], 3)
    assert_equal(frame.data.data[2], 4)

    frame.metadata.msg_version = 32
    assert_equal(frame.metadata.msg_version, 32)

    frame.metadata.attributes.resize(1)
    frame.metadata.attributes[0].data = "some data"
    frame.metadata.attributes[0].name = "some name"
    assert_equal(frame.metadata.attributes[0].data, "some data")
    assert_equal(frame.metadata.attributes[0].name, "some name")

    frame.metadata.err_values[0].type = "error_UNDEFINED"
    frame.metadata.err_values[0].value = 5.0
    assert_equal(len(frame.metadata.err_values), 1)
    assert_equal(frame.metadata.err_values[0].type, "error_UNDEFINED")
    assert_equal(frame.metadata.err_values[0].value, 5.0)

    frame.metadata.mode = "mode_GRAY"
    assert_equal(frame.metadata.mode, "mode_GRAY")

    frame.metadata.pixel_coeffs[0] = 10
    assert_equal(frame.metadata.pixel_coeffs[0], 10)

    frame.metadata.pixel_model = "pix_DISP"
    assert_equal(frame.metadata.pixel_model, "pix_DISP")

    frame.metadata.received_time.microseconds = 11
    assert_equal(frame.metadata.received_time.microseconds, 11)

    frame.metadata.status = "status_VALID"
    assert_equal(frame.metadata.status, "status_VALID")

    frame.metadata.time_stamp.microseconds = 12
    assert_equal(frame.metadata.time_stamp.microseconds, 12)

    frame.extrinsic.msg_version = 13
    assert_equal(frame.extrinsic.msg_version, 13)

    frame.extrinsic.has_fixed_transform = True
    assert_true(frame.extrinsic.has_fixed_transform)

    frame.extrinsic.pose_robot_frame_sensor_frame.data.translation.fromarray(
        np.array([1.0, 2.0, 3.0]))
    assert_array_equal(
        frame.extrinsic.pose_robot_frame_sensor_frame.data.translation.toarray(),
        np.array([1.0, 2.0, 3.0]))

    frame.extrinsic.pose_fixed_frame_robot_frame.data.translation.fromarray(
        np.array([1.0, 2.0, 3.0]))
    assert_array_equal(
        frame.extrinsic.pose_fixed_frame_robot_frame.data.translation.toarray(),
        np.array([1.0, 2.0, 3.0]))

    frame.intrinsic.msg_version = 5
    assert_equal(frame.intrinsic.msg_version, 5)

    frame.intrinsic.camera_matrix.fromarray(np.eye(3))
    assert_array_equal(
        frame.intrinsic.camera_matrix.toarray(), np.eye(3))

    frame.intrinsic.camera_model = "cam_PINHOLE"
    assert_equal(frame.intrinsic.camera_model, "cam_PINHOLE")

    frame.intrinsic.dist_coeffs[0] = 11
    assert_equal(frame.intrinsic.dist_coeffs[0], 11)

    frame.intrinsic.sensor_id = "camera5"
    assert_equal(frame.intrinsic.sensor_id, "camera5")


def test_framepair():
    framepair = cdff_types.FramePair()

    framepair.msg_version = 7
    assert_equal(framepair.msg_version, 7)

    framepair.baseline = 10.0
    assert_equal(framepair.baseline, 10.0)

    framepair.left.msg_version = 41
    framepair.right.msg_version = 51

    assert_equal(framepair.left.msg_version, 41)
    assert_equal(framepair.right.msg_version, 51)


def test_framepair_str_to_yaml_smoke():
    framepair = cdff_types.FramePair()
    yaml.load(str(framepair))


def test_map():
    map = cdff_types.Map()

    assert_equal(map.msg_version, 0)
    map.msg_version = 42
    assert_equal(map.msg_version, 42)

    map.metadata.msg_version = 43
    assert_equal(map.metadata.msg_version, 43)

    map.metadata.time_stamp.microseconds = 11
    assert_equal(map.metadata.time_stamp.microseconds, 11)

    map.metadata.type = "map_DEM"
    assert_equal(map.metadata.type, "map_DEM")

    map.metadata.err_values[0].type = "error_UNDEFINED"
    map.metadata.err_values[0].value = 5.0
    assert_equal(len(map.metadata.err_values), 1)
    assert_equal(map.metadata.err_values[0].type, "error_UNDEFINED")
    assert_equal(map.metadata.err_values[0].value, 5.0)

    map.metadata.scale = 55.0
    assert_equal(map.metadata.scale, 55.0)

    map.metadata.pose_fixed_frame_map_frame.data.translation.fromarray(
        np.array([1.2, 3.4, 5.6]))
    assert_array_equal(map.metadata.pose_fixed_frame_map_frame.data.translation,
                       np.array([1.2, 3.4, 5.6]))

    assert_equal(map.data.msg_version, 1)
    map.data.msg_version = 44
    assert_equal(map.data.msg_version, 44)

    map.data.rows = 25
    assert_equal(map.data.rows, 25)

    map.data.cols = 26
    assert_equal(map.data.cols, 26)

    map.data.channels = 27
    assert_equal(map.data.channels, 27)

    map.data.depth = "depth_8U"
    assert_equal(map.data.depth, "depth_8U")

    map.data.row_size = 28
    assert_equal(map.data.row_size, 28)

    map.data.data[0] = 10
    map.data.data[1] = 11
    map.data.data[2] = 12
    map.data.data[3] = 13
    assert_equal(len(map.data.data), 4)
    assert_equal(map.data.data[0], 10)
    assert_equal(map.data.data[1], 11)
    assert_equal(map.data.data[2], 12)
    assert_equal(map.data.data[3], 13)

    assert_equal(
        str(map),
        "{type: Map, metadata: {time_stamp: {type: Time, microseconds: 11}, "
        "type: map_DEM, err_values: [{type: error_UNDEFINED, value: 5.0}], "
        "scale: 55, pose_fixed_frame_map_frame: {metadata: "
        "{msg_version: 0, producer_id: , parent_frame_id: , "
        "parent_time: {type: Time, microseconds: -9223372036854775807}, "
        "child_frame_id: , child_time: {type: Time, microseconds: "
        "-9223372036854775807}}, data: {translation: {type: Vector3d, "
        "data: [1.2, 3.4, 5.6]}, orientation: {type: Quaterniond, data: "
        "{x: 0, y: 0, z: 0, w: 0}}, cov: {type: Matrix6d, data: "
        "[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], "
        "[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]}}}}, "
        "data: {rows: 25, cols: 26, channels: 27, depth: depth_8U, "
        "row_size: 28}}")


def test_map_array_reference():
    m = cdff_types.Map()
    m.metadata.scale = 0.1
    m.data.rows = 100
    m.data.cols = 100
    m.data.channels = 1
    m.data.row_size = 0
    m.data.depth = "depth_32F"
    data = m.data.array_reference()
    assert_array_equal(data.shape, (100, 100, 1))
    assert_equal(data.dtype, np.float32)


def test_deserialize_uper():
    with open("test/test_data/pose.uper", "rb") as f:
        uper = pickle.load(f)
    pose = cdff_types.TransformWithCovariance()
    pose.from_uper(uper)
    assert_equal(pose.metadata.msg_version, 1)
    assert_equal(pose.metadata.producer_id, "")
    assert_equal(pose.metadata.parent_frame_id, "LocalTerrainFrame")
    assert_equal(pose.metadata.parent_time.microseconds, 1540374075138837)
    assert_equal(pose.metadata.child_frame_id, "RoverBodyFrame")
    assert_equal(pose.metadata.child_time.microseconds, 1540374075138837)
    assert_array_almost_equal(
        pose.data.translation.toarray(), [-12.281598, -28.618572, -0.083111])
    assert_array_almost_equal(
        pose.data.orientation.toarray(),
        [-0.012058, 0.010269, -0.970555, 0.24036])
    assert_equal(pose.data.cov.toarray().sum(), 172773508.99806702)
