from cdff_dev import logloader, typefromdict
import cdff_types
from nose.tools import assert_raises_regex, assert_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal


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
         "colors": [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
         "intensity": [2, 3, 4]}
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
    assert_equal(obj.data.intensity[0], 2)
    assert_equal(obj.data.intensity[1], 3)
    assert_equal(obj.data.intensity[2], 4)


def test_create_image():
    log = logloader.load_log("test/test_data/logs/frames2.msg")
    data = log["/hcru0/pt_stereo_rect/left/image"][1]
    obj = typefromdict.create_from_dict(
        log["/hcru0/pt_stereo_rect/left/image.meta"]["type"], data)
    assert_equal(obj.data.depth, "depth_8U")
    assert_equal(obj.data.rows, 772)
    assert_equal(obj.data.cols, 1032)
    assert_equal(obj.metadata.mode, "mode_GRAY")
    assert_equal(obj.metadata.status, "status_VALID")
    assert_equal(obj.metadata.time_stamp.microseconds, 1532433037838941812)
    assert_equal(obj.metadata.received_time.microseconds, 1532433037906064777)


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


def test_create_gps_solution_with_incorect_timestamp_name():
    data = {'time': {'microsecond': 1478374858471759}}
    assert_raises_regex(
        ValueError,
        "Type '<class 'cdff_types.Time'>' has no field with name 'microsecond'",
        typefromdict.create_from_dict, "/gps/Solution", data)


def test_create_gps_solution_with_incorrect_field_type():
    data = {'position': "bla"}
    assert_raises_regex(
        TypeError, "Failed to set JointState.position = bla, "
                   "error message: a float is required",
        typefromdict.create_from_dict, "JointState", data)


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


def test_load_asn1_bitstream():
    log = logloader.load_log("test/test_data/logs/asn1_bitstream.msg")
    asn1_bitstream = log["/pom_pose"][0]
    typename = log["/pom_pose.meta"]["type"]
    assert_equal(asn1_bitstream["type"], "asn1SccTransformWithCovariance")
    obj = typefromdict.create_from_dict(typename, asn1_bitstream)
    assert_equal(obj.metadata.msg_version, 1)
    assert_equal(obj.metadata.producer_id, "")
    assert_equal(obj.metadata.parent_frame_id, "LocalTerrainFrame")
    assert_equal(obj.metadata.parent_time.microseconds, 1540374075138837)
    assert_equal(obj.metadata.child_frame_id, "RoverBodyFrame")
    assert_equal(obj.metadata.child_time.microseconds, 1540374075138837)
    assert_array_almost_equal(
        obj.data.translation.toarray(), [-12.281598, -28.618572,  -0.083111])
    assert_array_almost_equal(
        obj.data.orientation.toarray(),
        [-0.0120578, 0.0102693, -0.970555, 0.24036])


def test_load_asn1_bitstream_with_invalid_serialization():
    asn1_bitstream = {
        "serialization_method": 1,
        "type": "asn1SccMap",
        "data": []
    }
    assert_raises_regex(
        NotImplementedError, "Cannot decode serialization method 1",
        typefromdict.create_from_dict, "asn1_bitstream", asn1_bitstream
    )


def test_load_asn1_bitstream_with_unsupported_type():
    asn1_bitstream = {
        "serialization_method": 0,
        "type": "asn1SccFrame",
        "data": []
    }
    assert_raises_regex(
        NotImplementedError, "Cannot decode type asn1SccFrame from uPER",
        typefromdict.create_from_dict, "asn1_bitstream", asn1_bitstream
    )
