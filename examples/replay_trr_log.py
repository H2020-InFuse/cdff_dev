import numpy as np
import glob
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

    """
    def groundTruthInput(self, data):
        if not self.ground_truth_initialized:
            self.start_pos = data.pos.toarray()
            self.start_orient = data.orient.toarray()

            t = cdff_types.RigidBodyState()
            t.pos.fromarray(self.start_pos)
            t.orient.fromarray(self.start_orient)
            t.source_frame = "dgps0"
            t.target_frame = "start"
            t.timestamp.microseconds = self._timestamp
            self._set_transform(t, frame_transformation=False)

            self.ground_truth_initialized = True

        t = cdff_types.RigidBodyState()
        p, q = data.pos.toarray(), data.orient.toarray()
        p, q = _subtract_pose(p, q, self.start_pos, self.start_orient)
        t.pos.fromarray(p)
        t.orient.fromarray(q)
        t.source_frame = "ground_truth"
        t.target_frame = "dgps0"
        t.timestamp.microseconds = self._timestamp
        self._set_transform(t, frame_transformation=False)
    """

    def groundTruthInput(self, data):  # ignored
        self._set_transform(data, frame_transformation=True)

    def bodyJointInput(self, data):
        # TODO not completely sure of whether it is a data transformation
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


def _subtract_pose(p1, q1, p2, q2):
    p = p1 - p2
    q = _quaternion_product(q1, _quaternion_inverse(q2))
    return p, q


def _quaternion_inverse(q):
    # quaternion conjugate is the inverse for unit quaternions
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quaternion_product(q1, q2):
    q = np.empty(4)
    q[:3] = q1[-1] * q2[:3] + q2[-1] * q1[:3] + np.cross(q1[:3], q2[:3])
    q[-1] = q1[-1] * q2[-1] - np.dot(q1[:3], q2[:3])
    return q


def main():
    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            #"/slam_converter.cloud": "body",
            "/slam_filter.output": "velodyne",
            #"/velodyne_lidar.laser_scans": "body",
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
            #("/dgps.imu_pose", "transformer.groundTruth"),
            ("transformer.odometryTrajectory", "result.odometryTrajectory"),
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
    #log_folder = "/home/dfki.uni-bremen.de/afabisch/logs/20180926-1521.1/"
    #log_folder = "/media/afabisch/TOSHIBA EXT/Datasets/20180927_sherpa_and_hcru_log/20180927-1752.2/"
    log_folder = "/media/afabisch/TOSHIBA EXT/Datasets/20180927_sherpa_and_hcru_log/20180927-1756.1/"
    log_iterator = logloader.replay_files(
        [
         sorted(glob.glob(log_folder + "sherpa_tt_mcs_Logger_InFuse.msg")),
         sorted(glob.glob(log_folder + "body_joint_Logger_InFuse.msg")),
         sorted(glob.glob(log_folder + "sherpa_tt_slam_Logger_InFuse.msg")),
         #sorted(glob.glob(log_folder + "dgps_Logger_InFuse.msg")),
         #sorted(glob.glob(log_folder + "velodyne_lidar_Logger_InFuse.msg")),
        ],
        stream_names=[
            "/mcs_sensor_processing.rigid_body_state_out",
            "/body_joint.body_joint_samples",
            "/slam_filter.output",
            #"/slam_converter.cloud",
            #"/dgps.imu_pose",
            #"/velodyne_lidar.laser_scans",
        ]
    )
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
