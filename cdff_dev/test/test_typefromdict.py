from cdff_dev import typefromdict
import cdff_types
from nose.tools import assert_raises_regex, assert_equal
from numpy.testing import assert_array_equal


def test_create_unknown_type():
    assert_raises_regex(
        ValueError, "has no Python bindings",
        typefromdict.create_cpp, "Unknown")


def test_create_vector2d():
    obj = typefromdict.create_cpp("Vector2d")
    assert_equal(type(obj), cdff_types.Vector2d)


def test_create_from_dict_vector2d():
    data = [0, 1]
    obj = typefromdict.create_from_dict("Vector2d", data)
    assert_equal(data[0], obj[0])
    assert_equal(data[1], obj[1])


def test_create_rigid_body_state():
    data = {
        "timestamp":
        {
            "microseconds": 5,
            "usecPerSec": 6,
        },
        "sourceFrame": "bla",
        "targetFrame": "blub",
        "pos": [0, 1, 2],
        "cov_position": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "orient": [0, 1, 2, 3],
        "cov_orientation": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "velocity": [2, 3, 4],
        "cov_velocity": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "angular_velocity": [3, 4, 5],
        "cov_angular_velocity": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    }
    obj = typefromdict.create_from_dict("RigidBodyState", data)

    assert_equal(obj.timestamp.microseconds, 5)
    assert_equal(obj.timestamp.usec_per_sec, 6)
    assert_equal(obj.pos[0], 0)
    assert_equal(obj.pos[1], 1)
    assert_equal(obj.pos[2], 2)
    assert_equal(obj.source_frame, "bla")
    assert_equal(obj.target_frame, "blub")
    assert_array_equal(obj.cov_position, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert_array_equal(obj.orient, [0, 1, 2, 3])
    assert_array_equal(obj.cov_orientation, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert_array_equal(obj.velocity, [2, 3, 4])
    assert_array_equal(obj.cov_velocity, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert_array_equal(obj.angular_velocity, [3, 4, 5])
    assert_array_equal(obj.cov_angular_velocity,
                       [[0, 1, 2], [3, 4, 5], [6, 7, 8]])


def test_create_laser_scan():
    data = {
        "ref_time":
        {
            "microseconds": 5,
            "usecPerSec": 6,
        },
        "start_angle": 0,
        "angular_resolution": 0.1,
        "speed": 5.0,
        "ranges": list(range(20)),
        "minRange": 15,
        "maxRange": 25,
        "remission": list(range(20))
    }
    obj = typefromdict.create_from_dict("LaserScan", data)
    assert_equal(obj.ref_time.microseconds, 5)
    assert_equal(obj.ref_time.usec_per_sec, 6)
    assert_equal(obj.start_angle, 0)
    assert_equal(obj.angular_resolution, 0.1)
    assert_equal(obj.speed, 5.0)
    for i in range(20):
        assert_equal(obj.ranges[i], i)
    assert_equal(obj.min_range, 15)
    assert_equal(obj.max_range, 25)
    for i in range(20):
        assert_equal(obj.remission[i], i)


def test_create_pointcloud():
    data = {
        "ref_time":
        {
            "microseconds": 5,
            "usecPerSec": 6,
        },
        "points": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        "colors": [[255, 255, 255, 255], [255, 255, 255, 255],
                   [255, 255, 255, 255]]
    }
    obj = typefromdict.create_from_dict("Pointcloud", data)
    assert_equal(obj.ref_time.microseconds, 5)
    assert_equal(obj.ref_time.usec_per_sec, 6)
    assert_array_equal(obj.points[0], [0, 1, 2])
    assert_array_equal(obj.points[1], [1, 2, 3])
    assert_array_equal(obj.points[2], [2, 3, 4])
    assert_array_equal(obj.colors[0], [255, 255, 255, 255])
    assert_array_equal(obj.colors[1], [255, 255, 255, 255])
    assert_array_equal(obj.colors[2], [255, 255, 255, 255])
