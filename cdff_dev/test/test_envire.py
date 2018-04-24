import cdff_envire
from nose.tools import assert_equal, assert_not_equal, \
    assert_false, assert_true, assert_greater, \
    assert_regexp_matches


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