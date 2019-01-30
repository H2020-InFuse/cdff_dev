"""
============
SherpaTT Log
============

In this example we will replay wheel odometry data and point clouds.
Note that it will take a while to generate the log index.

The log data is available from Zenodo at

    TODO
"""
import os
from cdff_dev import (logloader, dataflowcontrol, envirevisualization,
                      transformer)
import cdff_types


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        transformer.EnvireDFN.__init__(self)

    def initialize_graph(self, graph):
        t = transformer.make_transform(
            translation=[0.0, 0.0, 0.442],
            orientation=[0.0, 0.0, 0.0, 1.0])
        graph.add_transform("body", "velodyne_plane_fixed", t)

        t = transformer.make_transform(
            translation=[-0.12889, -0.01697, 0.09081],
            orientation=[0.0, 0.0, 0.06540328, 0.99785891])
        graph.add_transform("velodyne_plane_moving", "velodyne", t)

        t = transformer.make_transform(
            translation=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0])
        graph.add_transform("config_sherpaTT_body", "body", t)

        t = transformer.make_transform(
            translation=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0])
        graph.add_transform("origin", "odometry", t)

    def wheelOdometryInput(self, data):
        self._set_transform(data, frame_transformation=False)

    def bodyJointInput(self, data):
        self._set_transform(data, frame_transformation=False)

    def odometryTrajectoryOutput(self):
        if self.graph_.contains_edge("odometry", "body"):
            position = cdff_types.Vector3d()
            body2odometry = self._get_transform(
                "odometry", "body", frame_transformation=True)
            position.fromarray(body2odometry.pos.toarray())
            return position
        else:
            return None


def replay_logfile_join(log_folder, logfiles):  # shortcut
    return logloader.replay_join([
        logloader.replay_logfile(
            os.path.join(log_folder, filename), stream_name)
        for filename, stream_name in logfiles
    ])


def main():
    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            "/slam_filter.output": "velodyne",
            "transformer.odometryTrajectory": "origin",
        },
        urdf_files=[],
        center_frame="odometry"
    )

    transformer = Transformer()
    transformer.set_configuration_file(
        "logs/open_hcru_bremen/recording_20180927-175146_sherpaTT_integration_tf.msg")
    dfc = dataflowcontrol.DataFlowControl(
        nodes={"transformer": transformer},
        connections=(
            ("/mcs_sensor_processing.rigid_body_state_out", "transformer.wheelOdometry"),
            ("/body_joint.body_joint_samples", "transformer.bodyJoint")
        ),
        periods={"transformer": 1},
        real_time=False,
        verbose=0
    )
    dfc.setup()

    from cdff_dev.diagrams import save_graph_png
    save_graph_png(dfc, "trr.png")

    log_folder = "logs/open_sherpatt_bremen"
    log_iterator = replay_logfile_join(
        log_folder,
        [("sherpa_tt_mcs_Logger_InFuse.msg", ["/mcs_sensor_processing.rigid_body_state_out"]),
         ("body_joint_Logger_InFuse.msg", ["/body_joint.body_joint_samples"]),
         ("sherpa_tt_slam_Logger_InFuse.msg", ["/slam_filter.output"])]
    )
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
