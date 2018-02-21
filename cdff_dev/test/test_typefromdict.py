from cdff_dev import typefromdict
from cdff_dev import cdff_types
from nose.tools import assert_raises_regex, assert_equal


def test_create_unknown_type():
    assert_raises_regex(
        ValueError, "has no Python bindings",
        typefromdict.create_cpp, "Unknown")


def test_create_vector2d():
    obj = typefromdict.create_cpp("Vector2d")
    assert_equal(type(obj), cdff_types.Vector2d)
