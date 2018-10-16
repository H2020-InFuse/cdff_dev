import os
import numpy as np
from cdff_dev import (logloader, dataflowcontrol, envirevisualization,
                      transformer)
import cdff_types
import cdff_envire


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()
        self.ground_truth_initialized = False
        self._graph = None

    def initialize_graph(self, graph):
        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.array([0.0, 0.0, 0.442]))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
        graph.add_transform("body", "velodyne_plane_fixed", t)

        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.array([-0.12889, -0.01697, 0.09081]))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.06540328, 0.99785891]))
        graph.add_transform("velodyne_plane_moving", "velodyne", t)


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


def main():
    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            "/slam_filter.output": "velodyne",
            "transformer.odometryTrajectory": "odometry",
        },
        urdf_files=[],
        center_frame="odometry"
    )

    dfc = dataflowcontrol.DataFlowControl(
        nodes={"transformer": Transformer()},
        connections=(
            ("/mcs_sensor_processing.rigid_body_state_out", "transformer.wheelOdometry"),
            ("/body_joint.body_joint_samples", "transformer.bodyJoint"),
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
    log_iterator = logloader.replay_join([
        logloader.replay_logfile(
            os.path.join(log_folder, "sherpa_tt_mcs_Logger_InFuse.msg"),
            ["/mcs_sensor_processing.rigid_body_state_out"]
        ),
        logloader.replay_logfile(
            os.path.join(log_folder, "body_joint_Logger_InFuse.msg"),
            ["/body_joint.body_joint_samples"]
        ),
        logloader.replay_logfile(
            os.path.join(log_folder, "sherpa_tt_slam_Logger_InFuse.msg"),
            ["/slam_filter.output"]
        )
    ])
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
