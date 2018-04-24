import cdff_envire
import numpy as np
from nose.tools import assert_equal, assert_not_equal, \
    assert_false, assert_true, assert_greater, \
    assert_regexp_matches, assert_raises_regexp, assert_almost_equal
from numpy.testing import assert_array_almost_equal


def test_get_set_microseconds():
    t = cdff_envire.Time()
    m = 1000023
    t.microseconds = m
    assert_equal(t.microseconds, m)


def test_time_str():
    t = cdff_envire.Time.now()
    assert_regexp_matches(str(t), "<time=\d{8}-\d{2}:\d{2}:\d{2}:\d{6}>")


def test_no_overflow():
    t = cdff_envire.Time.now()
    mus = t.microseconds
    assert_greater(mus, 0)
    t.microseconds = mus


def test_time_operators():
    t1 = cdff_envire.Time()
    t1.microseconds = 0
    t2 = cdff_envire.Time()
    t2.microseconds = 1
    t3 = cdff_envire.Time()
    t3.microseconds = 1
    t4 = cdff_envire.Time()
    t4.microseconds = 2

    assert_true(t1 < t2 < t4)
    assert_true(t4 > t2 > t1)
    assert_true(t1 != t2)
    assert_false(t2 != t3)
    assert_true(t2 == t3)
    assert_false(t1 == t2)
    assert_true(t2 >= t3)
    assert_false(t2 > t3)
    assert_true(t2 <= t3)
    assert_false(t2 < t3)

    assert_true(t2 + t2 == t4)
    assert_true(t4 - t2 == t2)
    assert_true(t4 // 2 == t2)
    assert_true(t2 * 2 == t4)

    t5 = cdff_envire.Time()
    t5.microseconds = 10
    t5 //= 2
    assert_equal(t5.microseconds, 5)
    t5 -= t2 * 4
    assert_equal(t5, t2)
    t5 *= 2
    assert_equal(t5, t4)
    t5 += t1
    assert_equal(t5, t4)


def test_time_assign():
    t1 = cdff_envire.Time()
    t2 = cdff_envire.Time.now()
    assert_not_equal(t1, t2)
    t1.assign(t2)
    assert_equal(t1, t2)


def test_vector3d_ctor():
    v = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    assert_equal(str(v), "[1.00, 2.00, 3.00]")


def test_vector3d_fromarray():
    v = cdff_envire.Vector3d()
    a = np.array([-2.32, 2.42, 54.23])
    v.fromarray(a)
    assert_equal(v[0], a[0])
    assert_equal(v[1], a[1])
    assert_equal(v[2], a[2])


def test_vector3d_as_ndarray():
    random_state = np.random.RandomState(843)
    r = random_state.randn(3, 3)
    v = cdff_envire.Vector3d(3.23, 2.24, 3.63)
    rv = r.dot(np.asarray(v))
    rv2 = r.dot(v.toarray())
    assert_array_almost_equal(rv, rv2)


def test_vector3d_get_set_data():
    v = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    v.x = 5.0
    v.y = 6.0
    v.z = 7.0
    assert_equal(v.x, 5.0)
    assert_equal(v.y, 6.0)
    assert_equal(v.z, 7.0)


def test_vector3d_array_access():
    v = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    assert_equal(v[0], 1.0)
    v[1] = 4.0
    assert_equal(v[1], 4.0)
    assert_raises_regexp(KeyError, "index must be", lambda i: v[i], -1)
    def assign(i):
        v[i] = 5.0
    assert_raises_regexp(KeyError, "index must be", assign, 3)


def test_vector3d_assign():
    obj1 = cdff_envire.Vector3d()
    obj2 = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    assert_not_equal(obj1, obj2)
    obj1.assign(obj2)
    assert_equal(obj1, obj2)


def test_norms():
    v = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    assert_almost_equal(v.norm(), 3.741657387)
    assert_equal(v.squared_norm(), 14.0)


def test_quaterniond_ctor():
    q = cdff_envire.Quaterniond(1.0, 0.0, 0.0, 0.0)
    assert_equal(str(q), "[im=1.00, real=(0.00, 0.00, 0.00)]")


def test_transform_with_cov_ctor():
    cdff_envire.TransformWithCovariance()


def test_transform_set_get_translation():
    p = cdff_envire.TransformWithCovariance()
    t = cdff_envire.Vector3d(1.0, 2.0, 3.0)
    p.translation = t
    assert_equal(str(p.translation), "[1.00, 2.00, 3.00]")


def test_transform_set_get_orientation():
    p = cdff_envire.TransformWithCovariance()
    o = cdff_envire.Quaterniond(1.0, 0.0, 0.0, 0.0)
    p.orientation = o
    assert_equal(str(p.orientation), "[im=1.00, real=(0.00, 0.00, 0.00)]")