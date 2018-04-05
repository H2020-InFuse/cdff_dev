from cdff_dev import logloader
from nose.tools import assert_in


def test_load_log():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    assert_in("/dynamixel.status_samples", log)
    assert_in("/dynamixel.status_samples.meta", log)
    assert_in("/dynamixel.transforms", log)
    assert_in("/dynamixel.transforms.meta", log)
    assert_in("/hokuyo.scans", log)
    assert_in("/hokuyo.scans.meta", log)
