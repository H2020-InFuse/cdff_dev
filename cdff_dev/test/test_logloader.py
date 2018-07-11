from cdff_dev import logloader
from nose.tools import (assert_in, assert_equal, assert_almost_equal,
                        assert_true, assert_raises)
import math
import glob


def test_load_log():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    assert_in("/dynamixel.status_samples", log)
    assert_in("/dynamixel.status_samples.meta", log)
    assert_in("/dynamixel.transforms", log)
    assert_in("/dynamixel.transforms.meta", log)
    assert_in("/hokuyo.scans", log)
    assert_in("/hokuyo.scans.meta", log)


def test_logfile_group_loads_sample():
    filenames = sorted(glob.glob("test/test_data/logs/xsens_imu_*.msg"))
    stream_names = ["/xsens_imu.calibrated_sensors"]
    lg = logloader.LogfileGroup(filenames, stream_names)

    assert_equal(lg.next_timestamp(), 1530195436455152)
    timestamp, stream_name, typename, sample = lg.next_sample()
    assert_equal(timestamp, 1530195436455152)
    assert_equal(stream_name, "/xsens_imu.calibrated_sensors")
    assert_equal(typename, "IMUSensors")
    assert_almost_equal(sample["acc"][0], -0.45818474888801575)


def test_logfile_group_loads_all_samples():
    filenames = sorted(glob.glob("test/test_data/logs/xsens_imu_*.msg"))
    stream_names = ["/xsens_imu.calibrated_sensors"]
    lg = logloader.LogfileGroup(filenames, stream_names)

    for _ in range(300):
        lg.next_timestamp()
        lg.next_sample()  # ignore output

    timestamp = lg.next_timestamp()
    assert_true(math.isinf(timestamp))
    assert_raises(StopIteration, lg.next_sample)


def test_replay_files():
    filenames = [sorted(glob.glob("test/test_data/logs/xsens_imu_*.msg")),
                 sorted(glob.glob("test/test_data/logs/dynamixel_*.msg"))]
    stream_names = ["/xsens_imu.calibrated_sensors",
                    "/dynamixel.transforms"]

    sample_counter = 0
    for _ in logloader.replay_files(filenames, stream_names):
        sample_counter += 1

    assert_equal(sample_counter, 300 + 300)