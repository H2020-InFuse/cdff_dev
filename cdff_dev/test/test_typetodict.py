from cdff_dev import typetodict
import cdff_types
import numpy as np
from nose.tools import assert_equal


def test_time_to_dict():
    time = cdff_types.Time()
    time.microseconds = 5
    d = typetodict.convert_to_dict(time)
    assert_equal(d["microseconds"], 5)


def test_vector2_to_dict():
    v = cdff_types.Vector2d()
    v[0] = 2.0
    v[1] = 3.0
    d = typetodict.convert_to_dict(v)
    assert_equal(d[0], 2.0)
    assert_equal(d[1], 3.0)


def test_vector3_to_dict():
    v = cdff_types.Vector3d()
    v[0] = 2.0
    v[1] = 3.0
    v[2] = 4.0
    d = typetodict.convert_to_dict(v)
    assert_equal(d[0], 2.0)
    assert_equal(d[1], 3.0)
    assert_equal(d[2], 4.0)


def test_vector4_to_dict():
    v = cdff_types.Vector4d()
    v[0] = 2.0
    v[1] = 3.0
    v[2] = 4.0
    v[3] = 5.0
    d = typetodict.convert_to_dict(v)
    assert_equal(d[0], 2.0)
    assert_equal(d[1], 3.0)
    assert_equal(d[2], 4.0)
    assert_equal(d[3], 5.0)


def test_vector6_to_dict():
    v = cdff_types.Vector6d()
    v[0] = 2.0
    v[1] = 3.0
    v[2] = 4.0
    v[3] = 5.0
    v[4] = 6.0
    v[5] = 7.0
    d = typetodict.convert_to_dict(v)
    assert_equal(d[0], 2.0)
    assert_equal(d[1], 3.0)
    assert_equal(d[2], 4.0)
    assert_equal(d[3], 5.0)
    assert_equal(d[4], 6.0)
    assert_equal(d[5], 7.0)


def test_vectorx_to_dict():
    v = cdff_types.VectorXd()
    v[0] = 2.0
    v[1] = 3.0
    d = typetodict.convert_to_dict(v)
    assert_equal(d[0], 2.0)
    assert_equal(d[1], 3.0)


def test_matrix2_to_dict():
    m = cdff_types.Matrix2d()
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[1, 0] = 3.0
    m[1, 1] = 4.0
    d = typetodict.convert_to_dict(m)
    assert_equal(d[0][0], 1.0)
    assert_equal(d[0][1], 2.0)
    assert_equal(d[1][0], 3.0)
    assert_equal(d[1][1], 4.0)


def test_matrix3_to_dict():
    m = cdff_types.Matrix3d()
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[0, 2] = 3.0
    m[1, 0] = 4.0
    m[1, 1] = 5.0
    m[1, 2] = 6.0
    m[2, 0] = 7.0
    m[2, 1] = 8.0
    m[2, 2] = 9.0
    d = typetodict.convert_to_dict(m)
    assert_equal(d[0][0], 1.0)
    assert_equal(d[0][1], 2.0)
    assert_equal(d[0][2], 3.0)
    assert_equal(d[1][0], 4.0)
    assert_equal(d[1][1], 5.0)
    assert_equal(d[1][2], 6.0)
    assert_equal(d[2][0], 7.0)
    assert_equal(d[2][1], 8.0)
    assert_equal(d[2][2], 9.0)


def test_matrix3_to_dict():
    m = cdff_types.Matrix6d()
    m.fromarray(np.eye(6))
    d = typetodict.convert_to_dict(m)
    for i in range(6):
        for j in range(6):
            if i == j:
                assert_equal(d[i][j], 1.0)
            else:
                assert_equal(d[i][j], 0.0)


def test_quaternion_to_dict():
    q = cdff_types.Quaterniond()
    d = typetodict.convert_to_dict(q)
    assert_equal(d[0], 0.0)
    assert_equal(d[1], 0.0)
    assert_equal(d[2], 0.0)
    assert_equal(d[3], 1.0)


def test_pose_to_dict():
    pose = cdff_types.Pose()
    pose.pos[0] = 1.0
    pose.pos[1] = 2.0
    pose.pos[2] = 3.0
    pose.orient[0] = 0.0
    pose.orient[1] = 0.0
    pose.orient[2] = 0.0
    pose.orient[3] = 1.0
    d = typetodict.convert_to_dict(pose)
    assert_equal(list, type(d["pos"]))
    assert_equal(d["pos"][0], 1.0)
    assert_equal(d["pos"][1], 2.0)
    assert_equal(d["pos"][2], 3.0)
    assert_equal(list, type(d["orient"]))
    assert_equal(d["orient"][0], 0.0)
    assert_equal(d["orient"][1], 0.0)
    assert_equal(d["orient"][2], 0.0)
    assert_equal(d["orient"][3], 1.0)
