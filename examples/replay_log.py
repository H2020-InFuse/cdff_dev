import sys
import numpy as np
from cdff_dev import logloader, typefromdict, dataflowcontrol, envirevisualization
import cdff_types
from PyQt4.QtCore import *
from PyQt4.QtGui import *


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


class Step:
    def __init__(self, stream_names, log, dfc):
        self.iterator = logloader.replay(stream_names, log, verbose=0)
        self.dfc = dfc

    def __call__(self):
        timestamp, stream_name, typename, sample = next(self.iterator)
        obj = typefromdict.create_from_dict(typename, sample)
        self.dfc.process_sample(
            timestamp=timestamp, stream_name=stream_name, sample=obj)


def main():
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
    frames = {
        "/hokuyo.scans": "upper_dynamixel",
        "laser_filter.filteredScan": "upper_dynamixel",
        "/dynamixel.transforms": "lower_dynamixel",
        "pointcloud_builder.pointcloud": "body"
    }
    vis = dataflowcontrol.EnvireVisualization(frames)
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods, vis)
    dfc.setup()

    log = logloader.load_log("test/test_data/logs/test_log.msg")
    stream_names = ["/hokuyo.scans", "/dynamixel.transforms"]

    app = QApplication(sys.argv)
    worker = envirevisualization.Worker(Step(stream_names, log, dfc))
    worker.start()
    win = envirevisualization.ReplayMainWindow(worker)
    win.show()
    app.aboutToQuit.connect(worker.quit)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
