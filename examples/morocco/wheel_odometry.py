import os
import numpy as np
from cdff_dev import logloader, envirevisualization, dataflowcontrol, transformer
import cdff_types
import cdff_envire
# TODO dependency: pytransform; git@github.com:rock-learning/pytransform.git
import pytransform.rotations as pr

# TODO configuration
log_folder = "logs/Sherpa"
dgps_logfile = "dgps.msg"
mcs_logfile = "sherpa_tt_mcs.msg"
utm_zone = 32  # TODO Utah 12, Germany 32, Morocco 29?
utm_north = True


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()
        self.imu_initialized = False
        self.start_pos = np.zeros(3)
        self.start_orient = cdff_types.Quaterniond()
        self.current_pos = np.zeros(3)
        self.current_orient = cdff_types.Quaterniond()

    def initialize_graph(self, graph):
        # Frames
        # ------
        # odometry - base frame of odometry, anchored in the world
        graph.add_frame("odometry")
        # body - robot's body, used in odometry log
        graph.add_frame("body")
        # origin - initial position of the robot according to odometry, same as
        #          odometry
        #graph.add_frame("origin") - already in the graph (center frame)
        # imu0 - imu sensor frame on the robot, use to connect origin to initial
        #        imu measurements
        graph.add_frame("imu0")
        # dgps0 - initial GPS position with axes aligned to north, west, up
        graph.add_frame("dgps0")

        # body - odometry, variable, updated continuously
        # origin - odometry, constant, known before start
        t = transformer.make_transform(
            translation=[0, 0, 0], orientation=[0.0, 0.0, 0.0, 1.0])
        graph.add_transform("origin", "odometry", t)
        # origin - imu0, constant, known before start
        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.array([-0.185, 0.3139, 0.04164]))
        R = pr.matrix_from_euler_zyx([0.44, 2.225, 0.0])
        imu2body_wxyz = pr.quaternion_from_matrix(R)
        imu2body_xyzw = np.hstack((imu2body_wxyz[1:], [imu2body_wxyz[0]]))
        t.transform.orientation.fromarray(imu2body_xyzw)
        graph.add_transform("imu0", "origin", t)

        # origin - dgps0, constant, known before start
        t = transformer.make_transform(
            translation=[0.3, 0.0, -0.53],
            orientation=[0.0, 0.0, 0.26067301, -0.96542715])
        graph.add_transform("dgps0", "origin", t)
        graph.add_transform("dgps", "body", t)

    def groundTruth2dgps0Input(self, data):
        if not self.imu_initialized:
            self.start_pos[:] = data.pos.toarray()
            self.start_orient.fromarray(data.orient.toarray())

            t = cdff_types.RigidBodyState()
            t.pos.fromarray(self.start_pos)
            t.orient = self.start_orient
            t.source_frame = "dgps0"
            t.target_frame = "start"
            t.timestamp.microseconds = self._timestamp
            self._set_transform(t, frame_transformation=False)

            self.imu_initialized = True

        t = cdff_types.RigidBodyState()
        self.current_pos[:] = data.pos.toarray()
        self.current_orient.fromarray(data.orient.toarray())
        p, q = _subtract_pose(self.current_pos, self.current_orient,
                              self.start_pos, self.start_orient)
        t.pos.fromarray(p)
        t.orient = q
        t.source_frame = "ground_truth"
        t.target_frame = "dgps0"
        t.timestamp.microseconds = self._timestamp
        self._set_transform(t, frame_transformation=False)

    def body2odometryInput(self, data):
        self._set_transform(data, frame_transformation=False)

    def dgps2originOutput(self):
        return self._get_transform(
            "dgps", "origin", frame_transformation=False)

    def groundTruth2originOutput(self):
        return self._get_transform(
            "ground_truth", "origin", frame_transformation=False)

    def process(self):
        super(Transformer, self).process()


def _subtract_pose(p1, q1, p2, q2):
    p = p1 - p2
    # quaternion conjugate is the inverse for unit quaternions
    q = q1 * q2.conjugate()
    return p, q


class EvaluationDFN:
    def __init__(self):
        self.odometry_timestamps_ = []
        self.odometry_positions_ = []
        self.ground_truth_timestamps_ = []
        self.ground_truth_positions_ = []
        self.error = 0.0

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def dgps2originInput(self, odometry_pose):
        if odometry_pose.timestamp.microseconds > 0:
            self.odometry_timestamps_.append(odometry_pose.timestamp.microseconds)
            self.odometry_positions_.append(odometry_pose.pos.toarray())

    def groundTruth2originInput(self, ground_truth_pose):
        if ground_truth_pose.timestamp.microseconds > 0:
            self.ground_truth_timestamps_.append(ground_truth_pose.timestamp.microseconds)
            self.ground_truth_positions_.append(ground_truth_pose.pos.toarray())

    def errorOutput(self):
        return self.error

    def process(self):
        if not self.odometry_positions_ or not self.ground_truth_positions_:
            return
        self.error = np.linalg.norm(self.odometry_positions_[-1] -
                                    self.ground_truth_positions_[-1])
        print("[Evaluation] position error: %.2f" % self.error)


class Trajectory:
    def __init__(self):
        self.rbs = None
        self.orient = np.array([0, 0, 0, 1], dtype=np.float)
        self.pos = np.zeros(3)
        self.output = cdff_types.Vector3d()

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def poseInput(self, data):
        self.rbs = data

    def posOutput(self):
        return self.output

    def process(self):
        if self.rbs is not None:
            self.orient[:] = self.rbs.orient.toarray()
            self.pos[:] = self.rbs.pos.toarray()

            for d in range(3):
                self.output[d] = self.pos[d]


def replay_logfile_join(log_folder, logfiles):  # shortcut
    return logloader.replay_join([
        logloader.replay_logfile(
            os.path.join(log_folder, filename), stream_name)
        for filename, stream_name in logfiles
    ])


def configure():
    nodes = {
        "transformer": Transformer(),
        "evaluation": EvaluationDFN(),
        "trajectory_gt": Trajectory(),
        "trajectory_dgps": Trajectory(),
    }
    periods = {
        # frequency of odometry: 0.01
        # frequency of gps: 0.05
        "transformer": 0.01,
        "evaluation": 1.0,
        "trajectory_dgps": 1.0,
        "trajectory_gt": 1.0,
    }
    connections = (
        # inputs to transformer
        ("/dgps.imu_pose", "transformer.groundTruth2dgps0"),
        ("/mcs_sensor_processing.rigid_body_state_out", "transformer.body2odometry"),

        # inputs to evaluation
        ("transformer.dgps2origin", "evaluation.dgps2origin"),
        ("transformer.groundTruth2origin", "evaluation.groundTruth2origin"),

        # inputs to trajectory
        ("transformer.groundTruth2origin", "trajectory_gt.pose"),
        ("transformer.dgps2origin", "trajectory_dgps.pose"),

        # outputs, this is necessary so register the output port in the dfc
        ("evaluation.error", "result.error"),
        ("trajectory_dgps.pos","result.trajectory_dgps"),
        ("trajectory_gt.pos","result.trajectory_gt"),
    )
    frames = {"trajectory_dgps.pos": "origin", "trajectory_gt.pos": "origin"}

    urdf_files = [
        # git clone git@git.hb.dfki.de:facilitators/bundle-sherpa_tt.git
        #"bundle-sherpa_tt/data/sherpa_tt.urdf"  # TODO uncomment
    ]

    log_iterator = replay_logfile_join(
        log_folder,
        [(mcs_logfile, ["/mcs_sensor_processing.rigid_body_state_out"]),
         (dgps_logfile, ["/dgps.imu_pose"])]
    )

    app = envirevisualization.EnvireVisualizerApplication(
        frames, urdf_files, center_frame="origin")
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=False)
    dfc.setup()

    app.show_controls(log_iterator, dfc)

    return app, dfc


def evaluate(dfc):
    odometry = dfc.nodes["evaluation"].odometry_positions_
    ground_truth = dfc.nodes["evaluation"].ground_truth_positions_
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(aspect="equal")
    plt.grid()
    plt.plot([p[0] for p in ground_truth], [p[1] for p in ground_truth],
             label="Ground Truth")
    plt.plot([p[0] for p in odometry], [p[1] for p in odometry],
             label="Odometry")
    plt.legend()

    plt.show()


def main():
    app, dfc = configure()

    from cdff_dev.diagrams import save_graph_png
    save_graph_png(dfc, "wheel_odometry.png")

    app.exec_()
    evaluate(dfc)


if __name__ == "__main__":
    main()
