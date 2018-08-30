from cdff_dev import logloader, testing
from nose.tools import (assert_in, assert_equal, assert_almost_equal,
                        assert_true, assert_raises, assert_less_equal)
import math
import glob
import tempfile
import os


def test_load_log():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    assert_in("/dynamixel.status_samples", log)
    assert_in("/dynamixel.status_samples.meta", log)
    assert_in("/dynamixel.transforms", log)
    assert_in("/dynamixel.transforms.meta", log)
    assert_in("/hokuyo.scans", log)
    assert_in("/hokuyo.scans.meta", log)


def test_summarize_logfile():
    typenames, n_samples = logloader.summarize_logfile(
        "test/test_data/logs/test_log.msg")
    assert_in("/hokuyo.scans", typenames)
    assert_in("/hokuyo.state", typenames)
    assert_in("/hokuyo.timestamp_estimator_status", typenames)
    assert_in("/dynamixel.transforms", typenames)
    assert_in("/dynamixel.state", typenames)
    assert_in("/dynamixel.act_cycle_time", typenames)
    assert_equal(typenames["/hokuyo.scans"], "LaserScan")
    assert_equal(typenames["/dynamixel.transforms"], "RigidBodyState")
    assert_equal(n_samples["/hokuyo.scans"], 3611)
    assert_equal(n_samples["/dynamixel.transforms"], 902)


def test_summarize_logfiles():
    filenames = [sorted(glob.glob("test/test_data/logs/xsens_imu_*.msg")),
                 sorted(glob.glob("test/test_data/logs/dynamixel_*.msg"))]
    typenames = logloader.summarize_logfiles(filenames)
    assert_in("/xsens_imu.calibrated_sensors", typenames)
    assert_in("/dynamixel.transforms", typenames)
    assert_equal(typenames["/xsens_imu.calibrated_sensors"], "IMUSensors")
    assert_equal(typenames["/dynamixel.transforms"], "RigidBodyState")


def test_chunk_log():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    chunks = logloader.chunk_log(log, "/hokuyo.scans", 1000)
    assert_equal(len(chunks), 4)
    assert_equal(len(chunks[0]["/hokuyo.scans"]), 1000)
    assert_equal(len(chunks[0]["/hokuyo.scans.meta"]["timestamps"]), 1000)
    assert_equal(len(chunks[1]["/hokuyo.scans"]), 1000)
    assert_equal(len(chunks[1]["/hokuyo.scans.meta"]["timestamps"]), 1000)
    assert_equal(len(chunks[2]["/hokuyo.scans"]), 1000)
    assert_equal(len(chunks[2]["/hokuyo.scans.meta"]["timestamps"]), 1000)
    assert_equal(len(chunks[3]["/hokuyo.scans"]), 611)
    assert_equal(len(chunks[3]["/hokuyo.scans.meta"]["timestamps"]), 611)


def test_save_chunks():
    tempdir = tempfile.gettempdir()
    dirname = os.path.join(tempdir, "cdff_log")
    with testing.EnsureCleanup(dirname):
        log = logloader.load_log("test/test_data/logs/test_log.msg")
        chunks = logloader.chunk_log(log, "/hokuyo.scans", 1000)
        filename_prefix = os.path.join(dirname, "test_log")
        logloader.save_chunks(chunks, filename_prefix)
        assert_true(os.path.exists(filename_prefix + "_0.msg"))
        assert_true(os.path.exists(filename_prefix + "_1.msg"))
        assert_true(os.path.exists(filename_prefix + "_2.msg"))
        assert_true(os.path.exists(filename_prefix + "_3.msg"))

        chunk = logloader.load_log(filename_prefix + "_0.msg")
        assert_equal(len(chunk["/hokuyo.scans"]), 1000)


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
    last_timestamp = float("-inf")
    for timestamp, _, _, _ in logloader.replay_files(filenames, stream_names):
        sample_counter += 1
        assert_less_equal(last_timestamp, timestamp)
        last_timestamp = timestamp

    assert_equal(sample_counter, 300 + 300)