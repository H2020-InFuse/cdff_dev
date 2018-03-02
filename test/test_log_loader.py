from cdff_dev import logloader, typefromdict
from nose.tools import assert_in, assert_equal
from numpy.testing import assert_array_less


def test_load_log():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    assert_in("/dynamixel.status_samples", log)
    assert_in("/dynamixel.status_samples.meta", log)
    assert_in("/dynamixel.transforms", log)
    assert_in("/dynamixel.transforms.meta", log)
    assert_in("/hokuyo.scans", log)
    assert_in("/hokuyo.scans.meta", log)


def test_replay():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    stream_names = ["/hokuyo.scans", "/dynamixel.transforms"]
    timestamps = []
    for timestamp, stream_name, typename, sample in logloader.replay(
            stream_names, log, verbose=0):
        timestamps.append(timestamp)
        obj = typefromdict.create_from_dict(typename, sample)
        assert_equal(type(obj).__module__, "cdff_types")
    assert_array_less(timestamps[:-1], timestamps[1:])
