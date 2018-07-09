import numpy as np
from cdff_dev import logloader, dataflowcontrol, envirevisualization
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


def main():
    log = logloader.load_log(
        #"test/test_data/logs/test_log.msg"
        #"infuse.msg"
        #"xsens.calibrated_sensors.msg"
        "open_day.msg"
    )
    logloader.print_stream_info(log)
    stream_names = [
        #"/hokuyo.scans", "/dynamixel.transforms",
        #"/xsens.calibrated_sensors"
        "/xsens_imu.calibrated_sensors",
        #"/velodyne.laser_scans",
        "/tilt_scan.pointcloud",
    ]

    nodes = {
        #"laser_filter": LaserFilterDummyDFN(),
        #"pointcloud_builder": PointcloudBuilderDummyDFN()
    }
    periods = {
        #"laser_filter": 0.025,
        #"pointcloud_builder": 0.1
    }
    connections = (
        #("/hokuyo.scans", "laser_filter.scanSample"),
        #("laser_filter.filteredScan", "pointcloud_builder.scan"),
        #("/dynamixel.transforms", "pointcloud_builder.transform"),
        #("pointcloud_builder.pointcloud", "result.pointcloud")
    )
    frames = {
        #"/hokuyo.scans": "upper_dynamixel",
        #"laser_filter.filteredScan": "upper_dynamixel",
        #"/dynamixel.transforms": "lower_dynamixel",
        #"pointcloud_builder.pointcloud": "body",

        #"/xsens.calibrated_sensors": "body",

        "/xsens_imu.calibrated_sensors": "body",        
        "/velodyne.laser_scans": "body",
        "/tilt_scan.pointcloud": "body",
    }
    urdf_files = [
        "test/test_data/model.urdf"
        #"SeekurJr/urdf/seekurjr.urdf"
    ]

    app = envirevisualization.EnvireVisualizerApplication(frames, urdf_files)

    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=True)
    dfc.setup()

    app.show_controls(logloader.replay(stream_names, log, verbose=0), dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
