import glob
from cdff_dev import (logloader, dataflowcontrol, envirevisualization,
                      transformer)


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()

    def upper_dynamixel2lower_dynamixelInput(self, data):
        self._set_transform(data)


def main():
    nodes = {
        "transformer": Transformer()
    }
    periods = {
        "transformer": 0.1
    }
    connections = (
        ("/dynamixel.transforms", "transformer.upper_dynamixel2lower_dynamixel")
    )
    frames = {
        "/laser_filter.filtered_scans": "upper_dynamixel",
        "/tilt_scan.pointcloud": "body",
        "/dynamixel.transforms": "body",
        "/velodyne.laser_scans": "velodyne"
    }
    urdf_files = [
        "examples/seekurjr.urdf"
    ]

    stream_names = [
        "/laser_filter.filtered_scans",
        "/tilt_scan.pointcloud",
        "/dynamixel.transforms",
        "/velodyne.laser_scans",
    ]

    app = envirevisualization.EnvireVisualizerApplication(
        frames, urdf_files, center_frame="body")

    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=True)
    dfc.setup()

    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
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
