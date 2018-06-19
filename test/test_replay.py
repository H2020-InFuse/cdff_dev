import numpy as np
from cdff_dev import logloader, typefromdict, dataflowcontrol, replay
import cdff_types
from nose.tools import assert_equal, assert_in
from numpy.testing import assert_array_less


def test_replay():
    log = logloader.load_log("test/test_data/logs/test_log.msg")
    stream_names = ["/hokuyo.scans", "/dynamixel.transforms",
                    "/dynamixel.status_samples"]
    timestamps = []
    for timestamp, stream_name, typename, sample in logloader.replay(
            stream_names, log, verbose=0):
        timestamps.append(timestamp)
        obj = typefromdict.create_from_dict(typename, sample)
        assert_equal(type(obj).__module__, "cdff_types")
    assert_array_less(timestamps[:-1], timestamps[1:])


class LaserFilterDummyDFN:
    def __init__(self):
        self.scanSample = cdff_types.LaserScan()

    def configure(self):
        pass

    def scanSampleInput(self, scanSample):
        self.scanSample = scanSample

    def process(self):
        # filter all values that are too far away from the median
        ranges = np.array([self.scanSample.ranges[i]
                           for i in range(self.scanSample.ranges.size())])
        med = np.median(ranges)
        for i in range(len(ranges)):
            if abs(ranges[i] - med) > 100:
                self.scanSample.ranges[i] = med

    def filteredScanOutput(self):
        return self.scanSample


class PointcloudBuilderDummyDFN:
    def __init__(self):
        self.pointcloud = cdff_types.Pointcloud()

    def configure(self):
        pass

    def scanInput(self, scan):
        self.scan = scan

    def transformInput(self, transform):
        self.transform = transform

    def process(self):
        pass

    def pointcloudOutput(self):
        return self.pointcloud


def test_feed_data_flow_control():
    nodes = {
        "laser_filter": LaserFilterDummyDFN(),
        "pointcloud_builder": PointcloudBuilderDummyDFN()
    }
    periods = {
        "laser_filter": 0.025,
        "pointcloud_builder": 0.1
    }
    connections = (
        ("/hokuyo.scans", "laser_filter.scanSample"),
        ("laser_filter.filteredScan", "pointcloud_builder.scan"),
        ("/dynamixel.transforms", "pointcloud_builder.transform"),
        ("pointcloud_builder.pointcloud", "result.pointcloud")
    )
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods)
    dfc.setup()
    vis = dataflowcontrol.NoVisualization()
    dfc.set_visualization(vis)

    log = logloader.load_log("test/test_data/logs/test_log.msg")
    stream_names = ["/hokuyo.scans", "/dynamixel.transforms"]
    replay.replay_and_process(dfc, log, stream_names)
    assert_in("laser_filter.filteredScan", vis.data)
    assert_in("pointcloud_builder.pointcloud", vis.data)
