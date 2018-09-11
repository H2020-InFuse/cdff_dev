import glob
from cdff_dev import (logloader, dataflowcontrol, envirevisualization,
                      transformer)


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()

    def upper_dynamixel2lower_dynamixelInput(self, data):
        self._set_transform(data)


def main():
    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            "/laser_filter.filtered_scans": "upper_dynamixel",
            "/tilt_scan.pointcloud": "body",
            "/velodyne.laser_scans": "velodyne",
        },
        urdf_files=["examples/seekurjr.urdf"],
        center_frame="body"
    )

    dfc = dataflowcontrol.DataFlowControl(
        nodes={"transformer": Transformer()},
        connections=(
            ("/dynamixel.transforms",
             "transformer.upper_dynamixel2lower_dynamixel"),
        ),
        #periods={"transformer": 0.1},
        trigger_ports={"transformer": "upper_dynamixel2lower_dynamixel"},
        real_time=True
    )
    dfc.setup()

    from cdff_dev.diagrams import save_graph_png
    save_graph_png(dfc, "transformer.png")

    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    log_iterator = logloader.replay_files(
        [sorted(glob.glob("logs/open_day/open_day_laser_filter_*.msg")),
         sorted(glob.glob("logs/open_day/open_day_tilt_scan_*.msg")),
         sorted(glob.glob("logs/open_day/open_day_dynamixel_*.msg")),
         sorted(glob.glob("logs/open_day/open_day_velodyne_*.msg"))],
        stream_names=[
            "/laser_filter.filtered_scans",
            "/tilt_scan.pointcloud",
            "/dynamixel.transforms",
            "/velodyne.laser_scans",
        ]
    )
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
