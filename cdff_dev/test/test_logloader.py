from cdff_dev import logloader, testing
from nose.tools import (assert_in, assert_equal, assert_almost_equal,
                        assert_true, assert_raises, assert_less_equal,
                        assert_dict_equal, assert_raises_regexp)
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


def test_extract_sample_index_negative():
    assert_raises(
        ValueError, logloader._extract_sample_from_logfile,
        "test/test_data/logs/test_log.msg", "/hokuyo.scans", -1)


def test_extract_sample_index_too_large():
    assert_raises(
        ValueError, logloader._extract_sample_from_logfile,
        "test/test_data/logs/test_log.msg", "/hokuyo.scans", 3611)


def test_extract_sample():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    actual_sample = log["/hokuyo.scans"][3]
    sample = logloader._extract_sample_from_logfile(
        "test/test_data/logs/test_log.msg", "/hokuyo.scans", 3)
    assert_dict_equal(actual_sample, sample)


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


def test_summarize_log():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    typenames, n_samples = logloader.summarize_log(log)
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


def test_replay_join():
    try:
        log_iterators = [
            logloader.replay_logfile(
                "test/test_data/logs/xsens_imu_00.msg",
                ["/xsens_imu.calibrated_sensors"]
            ),
            logloader.replay_logfile(
                "test/test_data/logs/dynamixel_0.msg",
                ["/dynamixel.transforms"]
            )
        ]
        log_iterator = logloader.replay_join(log_iterators)
        stream_counter = {key: 0 for key in ["/xsens_imu.calibrated_sensors",
                                             "/dynamixel.transforms"]}
        last_timestamp = float("-inf")
        for t, sn, tn, s in log_iterator:
            stream_counter[sn] += 1
            assert_less_equal(last_timestamp, t)
            last_timestamp = t
        for counter in stream_counter.values():
            assert_equal(counter, 100)
    finally:
        for filename in ["test/test_data/logs/xsens_imu_00.msg.cdff_idx",
                         "test/test_data/logs/dynamixel_0.msg.cdff_idx"]:
            if os.path.exists(filename):
                os.remove(filename)


def test_build_index():
    logfile = "test/test_data/logs/xsens_imu_00.msg"
    indexfile = "test/test_data/logs/xsens_imu_00.msg.cdff_idx"
    sn = "/xsens_imu.calibrated_sensors"
    msn = "/xsens_imu.calibrated_sensors.meta"
    with logloader.mmap_readfile(logfile) as m:
        try:
            current_positions, metadata = logloader.build_index(logfile, m)
            assert_true(os.path.exists(indexfile))
            assert_equal(current_positions[sn], 34)
            assert_equal(metadata[msn]["type"], "IMUSensors")
            assert_equal(len(metadata[msn]["timestamps"]), 100)

            # loading previously created index...
            current_positions, metadata = logloader.build_index(logfile, m)
            assert_equal(current_positions[sn], 34)
            assert_equal(metadata[msn]["type"], "IMUSensors")
            assert_equal(len(metadata[msn]["timestamps"]), 100)
        finally:
            if os.path.exists(indexfile):
                os.remove(indexfile)


def test_chunk_replay_log():
    filename = "test/test_data/logs/test_log.msg"
    stream = "/dynamixel.transforms"
    logloader.chunk_and_save_logfile(filename, stream, 100)
    try:
        log = logloader.load_log(filename)
        full_iterator = logloader.replay([stream], log)
        chunk_iterator = logloader.replay_files(
            [sorted(glob.glob("test/test_data/logs/test_log_*.msg"))],
            [stream])
        while True:
            try:
                t_actual, sn_actual, tn_actual, s_actual = next(full_iterator)
                t, sn, tn, s = next(chunk_iterator)
                assert_equal(t, t_actual)
                assert_equal(sn, sn_actual)
                assert_equal(tn, tn_actual)
                assert_equal(s["timestamp"]["microseconds"],
                             s_actual["timestamp"]["microseconds"])
                assert_equal(s["pos"], s_actual["pos"])
                assert_equal(s["orient"], s_actual["orient"])
            except StopIteration:
                assert_raises(StopIteration, next, full_iterator)
                assert_raises(StopIteration, next, chunk_iterator)
                break
    finally:
        filenames = glob.glob("test/test_data/logs/test_log_*.msg")
        for filename in filenames:
            os.remove(filename)


def test_replay_filename():
    filename = "test/test_data/logs/test_log.msg"
    streams = ["/dynamixel.transforms", "/hokuyo.scans"]
    log_iterator = logloader.replay_logfile(filename, streams)
    stream_counter = {key: 0 for key in streams}
    log = logloader.load_log(filename)
    actual_iterator = logloader.replay(streams, log)
    for t, sn, tn, s in log_iterator:
        stream_counter[sn] += 1
        t_actual, sn_actual, tn_actual, s_actual = next(actual_iterator)
        assert_equal(t, t_actual)
        assert_equal(sn, sn_actual)
        assert_equal(tn, tn_actual)
        if sn == "/dynamixel.transforms":
            assert_equal(s["timestamp"]["microseconds"],
                         s_actual["timestamp"]["microseconds"])
            assert_equal(s["pos"], s_actual["pos"])
            assert_equal(s["orient"], s_actual["orient"])
            assert_equal(s["sourceFrame"], s_actual["sourceFrame"])
            assert_equal(s["targetFrame"], s_actual["targetFrame"])
        else:
            assert_equal(s["ranges"], s_actual["ranges"])
            assert_equal(s["ref_time"]["microseconds"],
                         s_actual["ref_time"]["microseconds"])
            assert_equal(s["speed"], s_actual["speed"])
            assert_equal(s["angular_resolution"], s_actual["angular_resolution"])
    for stream_name in streams:
        assert_equal(len(log[stream_name]), stream_counter[stream_name])


def test_group_pattern():
    filenames = logloader.group_pattern(
        "test/test_data/logs/", "xsens_imu_*.msg")
    assert_equal(
        filenames,
        ["test/test_data/logs/xsens_imu_00.msg",
         "test/test_data/logs/xsens_imu_01.msg",
         "test/test_data/logs/xsens_imu_02.msg"])


def test_group_pattern_directory_not_found():
    assert_raises_regexp(
        ValueError, "Directory .* does not exist", logloader.group_pattern,
        "does_not_exist/xsens_imu_", "*.msg")


def test_group_pattern_files_not_found():
    assert_raises_regexp(
        ValueError, "Could not find any files matching '.*', only found.*",
        logloader.group_pattern, "test/test_data/logs/does_not_exist",
        "_*.msg")


def test_replay_sequence_empty():
    log_iterator = logloader.replay_sequence([])
    assert_raises_regexp(
        ValueError, "Expected at least one log iterator",
        next, log_iterator)


def test_replay_sequence():
    log_iterator = logloader.replay_sequence([
        logloader.replay_logfile(filename, ["/xsens_imu.calibrated_sensors"])
        for filename in logloader.group_pattern(
            "test/test_data/logs/", "xsens_imu_*.msg")
    ])
    n_samples = 0
    for _, _, _, _ in log_iterator:
        n_samples += 1
    assert_equal(n_samples, 300)


def test_replay_logfile_sequence():
    log_iterator = logloader.replay_logfile_sequence(
        logloader.group_pattern("test/test_data/logs/", "xsens_imu_*.msg"),
        ["/xsens_imu.calibrated_sensors"])
    n_samples = 0
    for _, _, _, _ in log_iterator:
        n_samples += 1
    assert_equal(n_samples, 300)
