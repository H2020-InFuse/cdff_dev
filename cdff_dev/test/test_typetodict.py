from cdff_dev import typetodict
import cdff_types
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