from cdff_dev import cdff_types
from nose.tools import assert_equal, assert_greater, assert_regexp_matches


def test_get_set_microseconds():
    t = cdff_types.Time()
    m = 1000023
    t.microseconds = m
    assert_equal(t.microseconds, m)


def test_time_str():
    t = cdff_types.Time()
    assert_regexp_matches(str(t), "{type: Time, microseconds: \d+, usec_per_sec: \d+}")


def test_vector2d():
    v = cdff_types.Vector2d()
    assert_equal(len(v), 2)
