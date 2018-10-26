from cdff_dev import data_export
from nose.tools import assert_equal, assert_in, assert_raises_regexp
from numpy.testing import assert_array_equal


def test_port_name():
    port_names = data_export.get_port_names("test/test_data/logs/test_log.msg")
    assert_equal(
        port_names,
        ["/dynamixel.act_cycle_time",
         "/dynamixel.state",
         "/dynamixel.status_samples",
         "/dynamixel.transforms",
         "/hokuyo.scans",
         "/hokuyo.state",
         "/hokuyo.timestamp_estimator_status",
        ]
    )


def test_object2dataframe():
    log = {
        "/component.port":
        [
            {"microseconds": 0},
            {"microseconds": 1},
            {"microseconds": 2},
            {"microseconds": 3},
        ],
        "/component.port.meta":
        {
            "type": "Time",
            "timestamps": [2, 3, 4, 5]
        }
    }
    df = data_export.object2dataframe(
        log, port="/component.port", whitelist=["acc", "gyro", "mag"])
    assert_array_equal(df.index, [2, 3, 4, 5])
    assert_array_equal(df["microseconds"].values, [0, 1, 2, 3])


def test_object2dataframe_timestamped():
    log = {
        "/component.port":
        [
            {
                "timestamp": {"microseconds": 2},
                "acc": [2.34, 3.45, 4.56],
                "gyro": [2.34, 3.45, 4.56],
                "mag": [2.34, 3.45, 4.56]
            },
            {
                "timestamp": {"microseconds": 3},
                "acc": [2.34, 3.45, 4.56],
                "gyro": [2.34, 3.45, 4.56],
                "mag": [2.34, 3.45, 4.56]
            },
        ],
        "/component.port.meta":
        {
            "type": "IMUSensors",
            "timestamps": [0, 1]
        }
    }
    df = data_export.object2dataframe(
        log, port="/component.port", whitelist=["acc", "gyro", "mag"])
    assert_array_equal(df.index, [2, 3])
    assert_array_equal(df["acc.0"].values, [2.34, 2.34])
    assert_array_equal(df["gyro.1"].values, [3.45, 3.45])
    assert_array_equal(df["mag.2"].values, [4.56, 4.56])


def test_join_labels():
    log = {
        "/component.port":
        [
            {
                "timestamp": {"microseconds": 2},
                "acc": [2.34, 3.45, 4.56],
                "gyro": [2.34, 3.45, 4.56],
                "mag": [2.34, 3.45, 4.56]
            },
            {
                "timestamp": {"microseconds": 3},
                "acc": [2.34, 3.45, 4.56],
                "gyro": [2.34, 3.45, 4.56],
                "mag": [2.34, 3.45, 4.56]
            },
        ],
        "/component.port.meta":
        {
            "type": "IMUSensors",
            "timestamps": [0, 1]
        }
    }
    df = data_export.object2dataframe(
        log, port="/component.port", whitelist=["acc", "gyro", "mag"])
    labels = [1, 0]
    df = data_export.join_labels(df, labels, "mylabel")
    assert_in("mylabel", df)
    assert_array_equal(df["mylabel"], [1, 0])


def test_join_labels_wrong_length():
    log = {
        "/component.port":
        [
            {
                "timestamp": {"microseconds": 2},
                "acc": [2.34, 3.45, 4.56],
                "gyro": [2.34, 3.45, 4.56],
                "mag": [2.34, 3.45, 4.56]
            },
            {
                "timestamp": {"microseconds": 3},
                "acc": [2.34, 3.45, 4.56],
                "gyro": [2.34, 3.45, 4.56],
                "mag": [2.34, 3.45, 4.56]
            },
        ],
        "/component.port.meta":
        {
            "type": "IMUSensors",
            "timestamps": [0, 1]
        }
    }
    df = data_export.object2dataframe(
        log, port="/component.port", whitelist=["acc", "gyro", "mag"])
    labels = [1]
    assert_raises_regexp(
        ValueError, "Number .* do not match", data_export.join_labels,
        df, labels, "mylabel")


def test_unpack_time():
    log = {
        "/component.port":
        [
            {"microseconds": 0},
            {"microseconds": 1},
            {"microseconds": 2},
            {"microseconds": 3},
        ],
        "/component.port.meta":
        {
            "type": "Time",
            "timestamps": [0, 1, 2, 3]
        }
    }
    converted_log = data_export.object2relational(log)
    assert_equal(log["/component.port"][0]["microseconds"],
                 converted_log["/component.port"]["microseconds"][0])
    assert_equal(log["/component.port"][1]["microseconds"],
                 converted_log["/component.port"]["microseconds"][1])
    assert_equal(log["/component.port"][2]["microseconds"],
                 converted_log["/component.port"]["microseconds"][2])
    assert_equal(log["/component.port"][3]["microseconds"],
                 converted_log["/component.port"]["microseconds"][3])


def test_unpack_imu():
    log = {
        "/component.port":
        [
            {
                "timestamp": {"microseconds": 0},
                "acc": [2.34, 3.45, 4.56],
                "gyro": [2.34, 3.45, 4.56],
                "mag": [2.34, 3.45, 4.56]
            },
            {
                "timestamp": {"microseconds": 1},
                "acc": [2.34, 3.45, 4.56],
                "gyro": [2.34, 3.45, 4.56],
                "mag": [2.34, 3.45, 4.56]
            },
        ],
        "/component.port.meta":
        {
            "type": "IMUSensors",
            "timestamps": [0, 1]
        }
    }
    converted_log = data_export.object2relational(
        log, whitelist=["acc", "gyro", "mag"])
    assert_equal(log["/component.port"][0]["acc"][0],
                 converted_log["/component.port"]["acc.0"][0])
    assert_equal(log["/component.port"][0]["acc"][1],
                 converted_log["/component.port"]["acc.1"][0])
    assert_equal(log["/component.port"][0]["acc"][2],
                 converted_log["/component.port"]["acc.2"][0])
    assert_equal(log["/component.port"][1]["acc"][0],
                 converted_log["/component.port"]["acc.0"][1])
    assert_equal(log["/component.port"][1]["acc"][1],
                 converted_log["/component.port"]["acc.1"][1])
    assert_equal(log["/component.port"][1]["acc"][2],
                 converted_log["/component.port"]["acc.2"][1])
