import numpy as np
from cdff_dev import cdff_types
from nose.tools import assert_equal, assert_greater, assert_regexp_matches
from numpy.testing import assert_array_equal, assert_array_almost_equal


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


def test_create_pointcloud():
    pcl = cdff_types.Pointcloud()

    pcl.ref_time.microseconds = 0
    assert_equal(pcl.ref_time.microseconds, 0)

    pcl.points.resize(100)
    point = pcl.points[0]
    point[0] = 1.0
    point[1] = 2.0
    point[2] = 3.0
    assert_equal(pcl.points.size(), 100)
    assert_array_equal(pcl.points[0].toarray(), (1.0, 2.0, 3.0))

    pcl.colors.resize(100)
    color = pcl.colors[0]
    color[0] = 255.0
    color[1] = 255.0
    color[2] = 255.0
    color[3] = 255.0
    assert_equal(pcl.colors.size(), 100)
    assert_array_equal(pcl.colors[0].toarray(), (255.0, 255.0, 255.0, 255.0))


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

    ls.remission.resize(100)
    ls.remission[0] = 1.0
    ls.remission[1] = 2.0
    ls.remission[2] = 3.0
    assert_equal(ls.remission.size(), 100)
    assert_array_equal(np.asarray([ls.remission[i] for i in range(3)]),
       (1.0, 2.0, 3.0))

    ls.ranges.resize(100)
    ls.ranges[0] = 1
    ls.ranges[1] = 2
    ls.ranges[2] = 3
    assert_equal(ls.ranges.size(), 100)
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
    assert_equal(
        str(rbs),
        "{type: RigidBodyStateg, {type: Time, microseconds: 0, usec_per_sec: 0}, "
        "sourceFrame=source, targetFrame=target, ...}")


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
