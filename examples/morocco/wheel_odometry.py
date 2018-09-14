import numpy as np
from cdff_dev import logloader, envirevisualization, dataflowcontrol, transformer
import cdff_types
import cdff_envire
# TODO dependency: pytransform; git@github.com:rock-learning/pytransform.git
import pytransform.rotations as pr

# TODO configuration
dgps_logfile = "logs/Sherpa/dgps_Logger.log"
mcs_logfile = "logs/Sherpa/sherpa_tt_mcs_Logger.log"
utm_zone = 32  # TODO Utah 12, Germany 32, Morocco 29?
utm_north = True


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()
        self.imu_initialized = False
        self.start_pos = np.zeros(3)

    def groundTruth2dgps0Input(self, data):
        if not self.imu_initialized:
            self.start_pos = data.pos.toarray()
            self.start_orient = data.orient.toarray()

            t = cdff_types.RigidBodyState()
            t.pos.fromarray(self.start_pos)
            t.orient.fromarray(self.start_orient)
            t.source_frame = "dgps0"
            t.target_frame = "start"
            t.timestamp.microseconds = self._timestamp
            self._set_transform(t, frame_transformation=False)

            self.imu_initialized = True

        t = cdff_types.RigidBodyState()
        p, q = data.pos.toarray(), data.orient.toarray()
        p, q = _subtract_pose(p, q, self.start_pos, self.start_orient)
        t.pos.fromarray(p)
        t.orient.fromarray(q)
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


def main():
    logs, stream_names = convert_logs()
    app, dfc = configure(logs, stream_names)

    from cdff_dev.diagrams import save_graph_png
    save_graph_png(dfc, "wheel_odometry.png")


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


def main():
    logs, stream_names = convert_logs()
    app, dfc = configure(logs, stream_names)

    from cdff_dev.diagrams import save_graph_png
    save_graph_png(dfc, "wheel_odometry.png")

    app.exec_()
    evaluate(dfc)


def convert_logs():
    # TODO convert from pocolog to msgpack
    # TODO find number of samples per stream
    # TODO chunk logfiles

    # logs/Sherpa/dgps.msg
    # - /dgps.imu_pose
    # logs/Sherpa/sherpa_tt_mcs.msg
    # - /mcs_sensor_processing.rigid_body_state_out

    #logfiles = [["logs/20180829_DFKI_Sherpa/dgps.msg"], ["logs/20180829_DFKI_Sherpa/sherpa_tt_mcs.msg"]]
    logfiles = [["logs/Sherpa/dgps.msg"], ["logs/Sherpa/sherpa_tt_mcs.msg"]]
    stream_names = [
       "/dgps.imu_pose",
       "/mcs_sensor_processing.rigid_body_state_out",
    ]

    return logfiles, stream_names


def configure(logs, stream_names):
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
        "evaluation": 1.0, # TODO
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

    log_iterator = logloader.replay_files(logs, stream_names, verbose=0)

    app = envirevisualization.EnvireVisualizerApplication(
        frames, urdf_files, center_frame="origin")
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=False)
    dfc.setup()

    # TODO simplify graph initialization

    graph = app.visualization.world_state_.graph_

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
    t = cdff_envire.Transform()
    t.transform.translation.fromarray(np.zeros(3))
    t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
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
    t = cdff_envire.Transform()

    # works for old logfiles:
    t.transform.translation.fromarray(np.array([0.3, 0.0, -0.53]))
    t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.26067301, -0.96542715]))

    #t.transform.translation.fromarray(np.array([0.0, 0.0, 0.0]))
    #t.transform.translation.fromarray(np.array([0.3, 0.0, -0.53]))
    #t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.26067301, -0.96542715]))
    #t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
    #t.transform.orientation.fromarray(np.array([  0.        ,  0.25881905,  0., 0.96592583]))
    #t.transform.orientation.fromarray(np.array([2.58819045e-01,   1.58480958e-17,   9.65925826e-01, -5.91458986e-17]))
    graph.add_transform("dgps0", "origin", t)
    graph.add_transform("dgps", "body", t)

    # TODO visualize covariance?

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


if __name__ == "__main__":
    main()
