from cdff_dev import logloader, typefromdict
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
        "metadata":
        {"timeStamp": {"microseconds": 5}},
        "data":
        {"points": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
         "colors": [[255, 255, 255], [255, 255, 255], [255, 255, 255]]}
    }
    obj = typefromdict.create_from_dict("Pointcloud", data)
    assert_equal(obj.metadata.time_stamp.microseconds, 5)
    assert_equal(obj.data.points[0, 0], 0)
    assert_equal(obj.data.points[0, 1], 1)
    assert_equal(obj.data.points[0, 2], 2)
    assert_equal(obj.data.points[1, 0], 1)
    assert_equal(obj.data.points[1, 1], 2)
    assert_equal(obj.data.points[1, 2], 3)
    assert_equal(obj.data.points[2, 0], 2)
    assert_equal(obj.data.points[2, 1], 3)
    assert_equal(obj.data.points[2, 2], 4)
    assert_equal(obj.data.colors[0, 0], 255)
    assert_equal(obj.data.colors[0, 1], 255)
    assert_equal(obj.data.colors[0, 2], 255)
    assert_equal(obj.data.colors[1, 0], 255)
    assert_equal(obj.data.colors[1, 1], 255)
    assert_equal(obj.data.colors[1, 2], 255)
    assert_equal(obj.data.colors[2, 0], 255)
    assert_equal(obj.data.colors[2, 1], 255)
    assert_equal(obj.data.colors[2, 2], 255)


def test_create_image():
    log = logloader.load_log("test/test_data/logs/frames.msg")
    data = log["/camera1.frame"][1]
    # HACK: Type is actually the old Frame
    obj = typefromdict.create_from_dict(
        log["/camera1.frame.meta"]["type"], data)
    assert_equal(obj.attributes[0].att_name, "FrameCount")
    assert_equal(obj.attributes[0].data, "86375")
    assert_equal(obj.attributes[1].att_name, "Exposure_ms")
    assert_equal(obj.attributes[1].data, "4.99986")
    assert_equal(obj.data_depth, 3)
    assert_equal(obj.datasize.height, 512)
    assert_equal(obj.datasize.width, 640)
    assert_equal(obj.frame_mode, "mode_rgb")
    assert_equal(obj.frame_status, "status_valid")
    assert_equal(obj.frame_time.microseconds, 1530195435880000)
    assert_equal(obj.received_time.microseconds, 1530195435905227)


def test_create_gps_solution():
    data = {
        'time': {'microseconds': 1478374858471759},
        'positionType': 'DIFFERENTIAL',
        'latitude': 38.41203398374147,
        'longitude': -110.78470430703186,
        'altitude': 1353.6719970703125,
        'ageOfDifferentialCorrections': 0.0,
        'noOfSatellites': 0,
        'geoidalSeparation': 0.0,
        'deviationLatitude': 0.023057984188199043,
        'deviationLongitude': 0.017021523788571358,
        'deviationAltitude': 0.0556030310690403
    }
    obj = typefromdict.create_from_dict("/gps/Solution", data)
    assert_equal(obj.time.microseconds, 1478374858471759)
    assert_equal(obj.position_type, "DIFFERENTIAL")
    assert_equal(obj.latitude, 38.41203398374147)
    assert_equal(obj.longitude, -110.78470430703186)
    assert_equal(obj.altitude, 1353.6719970703125)
    assert_equal(obj.age_of_differential_corrections, 0.0)
    assert_equal(obj.no_of_satellites, 0)
    assert_equal(obj.geoidal_separation, 0.0)
    assert_equal(obj.deviation_latitude, 0.023057984188199043)
    assert_equal(obj.deviation_longitude, 0.017021523788571358)
    assert_equal(obj.deviation_altitude, 0.0556030310690403)


def test_create_map():
    data = {
        "msgVersion": 1,
        "metadata": {
            "msgVersion": 1,
            "timeStamp": {
                "microseconds": 5
            },
            "type": "map_NAV",
            "err_values": [
                {"type": "error_UNDEFINED", "value": 0.0},
                {"type": "error_DEAD", "value": 1.0},
                {"type": "error_FILTERED", "value": 2.0}
            ],
            "scale": 2.0,
            "pose_fixed_frame_map_frame": {
                "data": {
                    "translation": [0.0, 1.0, 2.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0]
                }
            }
        },
        "data": {
            "msgVersion": 1,
            "rows": 2,
            "cols": 2,
            "depth": "depth_8U",
            "row_size": 2,
            "data": [0.0, 0.0, 1.0, 0.0]
        },
    }
    obj = typefromdict.create_from_dict("Map", data)
    assert_equal(obj.msg_version, 1)
    assert_equal(obj.metadata.msg_version, 1)
    assert_equal(obj.metadata.time_stamp.microseconds, 5)
    assert_equal(obj.metadata.type, "map_NAV")
    assert_equal(obj.metadata.err_values[0].type, "error_UNDEFINED")
    assert_equal(obj.metadata.err_values[0].value, 0.0)
    assert_equal(obj.metadata.err_values[1].type, "error_DEAD")
    assert_equal(obj.metadata.err_values[1].value, 1.0)
    assert_equal(obj.metadata.err_values[2].type, "error_FILTERED")
    assert_equal(obj.metadata.err_values[2].value, 2.0)
    assert_equal(obj.metadata.scale, 2.0)
    assert_equal(
        obj.metadata.pose_fixed_frame_map_frame.data.translation[0], 0.0)
    assert_equal(
        obj.metadata.pose_fixed_frame_map_frame.data.translation[1], 1.0)
    assert_equal(
        obj.metadata.pose_fixed_frame_map_frame.data.translation[2], 2.0)
    assert_equal(
        obj.metadata.pose_fixed_frame_map_frame.data.orientation[0], 0.0)
    assert_equal(
        obj.metadata.pose_fixed_frame_map_frame.data.orientation[1], 0.0)
    assert_equal(
        obj.metadata.pose_fixed_frame_map_frame.data.orientation[2], 0.0)
    assert_equal(
        obj.metadata.pose_fixed_frame_map_frame.data.orientation[3], 1.0)
    assert_equal(obj.data.msg_version, 1)
    assert_equal(obj.data.rows, 2)
    assert_equal(obj.data.cols, 2)
    assert_equal(obj.data.depth, "depth_8U")
    assert_equal(obj.data.row_size, 2)
    assert_equal(obj.data.data[0], 0.0)
    assert_equal(obj.data.data[1], 0.0)
    assert_equal(obj.data.data[2], 1.0)
    assert_equal(obj.data.data[3], 0.0)