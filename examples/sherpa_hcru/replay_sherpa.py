import os
import numpy as np
from cdff_dev import (logloader, dataflowcontrol, envirevisualization,
                      transformer)
import cdff_types
import cdff_envire


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        transformer.EnvireDFN.__init__(self)
        self.imu_initialized = False

    def initialize_graph(self, graph):
        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.array([0.0, 0.0, 0.442]))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
        graph.add_transform("body", "velodyne_plane_fixed", t)

        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.array([-0.12889, -0.01697, 0.09081]))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.06540328, 0.99785891]))
        graph.add_transform("velodyne_plane_moving", "velodyne", t)

        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.zeros(3))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
        graph.add_transform("config_sherpaTT_body", "body", t)

        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.zeros(3))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
        graph.add_transform("origin", "odometry", t)

        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.array([0.3, 0.0, -0.53]))
        # turned by 180 degrees around z-axis
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.70682518,  0.70738827]))

        graph.add_transform("dgps0", "origin", t)
        graph.add_transform("dgps", "body", t)

    def groundTruthInput(self, data):
        if not self.imu_initialized:
            t = cdff_types.RigidBodyState()
            t.pos.fromarray(data.pos.toarray())
            t.orient.fromarray(data.orient.toarray())
            t.source_frame = "dgps0"
            t.target_frame = "start"
            t.timestamp.microseconds = self._timestamp
            self._set_transform(t, frame_transformation=False)

            self.imu_initialized = True

        t = cdff_types.RigidBodyState()
        t.pos.fromarray(data.pos.toarray())
        t.orient.fromarray(data.orient.toarray())
        t.source_frame = "ground_truth"
        t.target_frame = "start"
        t.timestamp.microseconds = self._timestamp
        self._set_transform(t, frame_transformation=False)

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

    def groundTruthTrajectoryOutput(self):
        try:
            position = cdff_types.Vector3d()
            ground_truth2origin = self._get_transform(
                "origin", "ground_truth", frame_transformation=True)
            position.fromarray(ground_truth2origin.pos.toarray())
            return position
        except:
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
            "transformer.groundTruthTrajectory": "origin",
        },
        urdf_files=[],
        center_frame="odometry"
    )

    transformer = Transformer()
    transformer.set_configuration_file(
        "logs/sherpa_hcru/recording_20180927-175146_sherpaTT_integration_tf.msg")
    dfc = dataflowcontrol.DataFlowControl(
        nodes={"transformer": transformer},
        connections=(
            ("/mcs_sensor_processing.rigid_body_state_out", "transformer.wheelOdometry"),
            ("/body_joint.body_joint_samples", "transformer.bodyJoint"),
            ("/dgps.imu_pose", "transformer.groundTruth")
        ),
        periods={"transformer": 1},
        real_time=False,
        verbose=0
    )
    dfc.setup()

    from cdff_dev.diagrams import save_graph_png
    save_graph_png(dfc, "trr.png")

    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    #log_folder = "logs/20180927-1752_sherpa"
    log_folder = "logs/20180927-1756_sherpa"
    log_iterator = replay_logfile_join(
        log_folder,
        [("sherpa_tt_mcs_Logger_InFuse.msg", ["/mcs_sensor_processing.rigid_body_state_out"]),
         ("body_joint_Logger_InFuse.msg", ["/body_joint.body_joint_samples"]),
         ("sherpa_tt_slam_Logger_InFuse.msg", ["/slam_filter.output"]),
         ("dgps_Logger_InFuse.msg", ["/dgps.imu_pose"])]
    )
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
