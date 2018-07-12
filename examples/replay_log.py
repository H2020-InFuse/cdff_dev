import numpy as np
import glob
from cdff_dev import (logloader, dataflowcontrol, envirevisualization,
                      transformer)
import cdff_types


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


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()

    def upper_dynamixel2lower_dynamixelInput(self, data):
        self._set_transform(data)


def main():
    nodes = {
        "transformer": Transformer(),
        #"laser_filter": LaserFilterDummyDFN(),
        #"pointcloud_builder": PointcloudBuilderDummyDFN()
    }
    periods = {
        "transformer": 0.1,
        #"laser_filter": 0.025,
        #"pointcloud_builder": 0.1
    }
    connections = (
        ("/dynamixel.transforms", "transformer.upper_dynamixel2lower_dynamixel"),
        #("/hokuyo.scans", "laser_filter.scanSample"),
        #("laser_filter.filteredScan", "pointcloud_builder.scan"),
        #("pointcloud_builder.pointcloud", "result.pointcloud")
    )
    frames = {
        #"/hokuyo.scans": "upper_dynamixel",
        #"laser_filter.filteredScan": "upper_dynamixel",
        #"/dynamixel.transforms": "lower_dynamixel",
        #"pointcloud_builder.pointcloud": "body",

        #"/xsens.calibrated_sensors": "body",

        "/laser_filter.filtered_scans": "upper_dynamixel",
        "/tilt_scan.pointcloud": "body",
        "/dynamixel.transforms": "body",
        "/velodyne.laser_scans": "velodyne",

        #"/xsens.calibrated_sensors": "body"
    }
    urdf_files = [
        #"test/test_data/model.urdf"
        #"SeekurJr/urdf/seekurjr.urdf"
        "examples/seekurjr.urdf"
    ]

    stream_names = [
        #"/hokuyo.scans", "/dynamixel.transforms",
        #"/xsens.calibrated_sensors"
        #"/xsens_imu.calibrated_sensors",
        #"/velodyne.laser_scans",
        #"/tilt_scan.pointcloud",

        "/laser_filter.filtered_scans",
        "/tilt_scan.pointcloud",
        "/dynamixel.transforms",
        "/velodyne.laser_scans",
    ]
    #log = logloader.load_log(
    #    "test/test_data/logs/test_log.msg"
    #    "infuse.msg"
    #    "xsens.calibrated_sensors.msg"
    #    "open_day.msg"
    #)
    #logloader.print_stream_info(log)
    #log_iterator = logloader.replay(stream_names, log, verbose=0)

    app = envirevisualization.EnvireVisualizerApplication(
        frames, urdf_files, center_frame="body")

    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=True)
    dfc.setup()

    log_iterator = logloader.replay_files(
        [sorted(glob.glob("logs/open_day/open_day_laser_filter_*.msg")),
         sorted(glob.glob("logs/open_day/open_day_tilt_scan_*.msg")),
         sorted(glob.glob("logs/open_day/open_day_dynamixel_*.msg")),
         sorted(glob.glob("logs/open_day/open_day_velodyne_*.msg"))],
        stream_names
    )
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
