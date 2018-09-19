import numpy as np
from cdff_dev import qtgui, envirevisualization, logloader, dataflowcontrol
import cdff_types
from nose.tools import assert_true


class LaserFilterDummyDFN:
    def __init__(self):
        self.scanSample = cdff_types.LaserScan()

    def set_configuration_file(self, filename):
        pass

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

    def set_configuration_file(self, filename):
        pass

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


def test_step():
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

    log = logloader.load_log("test/test_data/logs/test_log.msg")
    stream_names = ["/hokuyo.scans", "/dynamixel.transforms"]

    step = qtgui.Step(logloader.replay(stream_names, log), dfc)
    stopped = False
    try:
        while True:
            step()
    except StopIteration:
        stopped = True
    assert_true(stopped)