"""
========================
Wheel Odometry vs. D-GPS
========================

This experiment with SherpaTT has been done in Morocco. SherpaTT
starts driving at a workshop towards a road to enter the desert.

We compare wheel odometry of SherpaTT with the D-GPS position which
represents ground truth in this experiment. The script will visualize
a simplified model of SherpaTT at the pose estimate derived from wheel
odometry, the trajectory of SherpaTT according to wheel odometry and
D-GPS, and a ground truth map with 4 cm resolution obtained from a
drone. Optionally we can show the joint movements of SherpaTT.
A simple DFN implemented in Python will compare both position
estimates and print the error between both to the terminal.

The log files and the GeoTIFF file for the ground truth map have been
uploaded to Zenodo:

    TODO

Copy them to the folder 'logs/open_morocco_wheel_odometry' to run
this script.
"""
import os
import numpy as np
from cdff_dev import logloader, envirevisualization, dataflowcontrol, \
    transformer, io
import cdff_types
import cdff_envire


# TODO configuration
log_folder = "logs/open_morocco_wheel_odometry"
dgps_logfile = "dgps_Logger_InFuse.msg"
odometry_logfile = "sherpa_tt_mcs_Logger_InFuse.msg"
joints_logfile = "sherpa_tt_mcs_Logger_Joints_InFuse.msg"
mapfile = "2018_11_27_xalucadunes_dsm.tif"
SHOW_ROBOT_MODEL = True
SHOW_JOINT_ANGLES = False


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()
        self.gps_initialized = False
        self.start_pos = np.zeros(3)
        self.start_orient = cdff_types.Quaterniond()
        self.current_pos = np.zeros(3)
        self.current_orient = cdff_types.Quaterniond()
        self.map_items = []

    def initialize_graph(self, graph):
        if SHOW_ROBOT_MODEL:
            self.urdf_model = cdff_envire.EnvireURDFModel()
            self.urdf_model.load_urdf(
                graph, "test/test_data/urdf_sherpa_meshes_LQ/sherpa_tt.urdf",
                load_visuals=True)

            beta = np.deg2rad(26.14)
            self.urdf_model.set_joint_angle("beta_front_left", -beta)
            self.urdf_model.set_joint_angle("beta1_fake_front_left", beta)
            self.urdf_model.set_joint_angle("beta2_fake_front_left", -beta)
            self.urdf_model.set_joint_angle("beta_front_right", -beta)
            self.urdf_model.set_joint_angle("beta1_fake_front_right", beta)
            self.urdf_model.set_joint_angle("beta2_fake_front_right", -beta)
            self.urdf_model.set_joint_angle("beta_rear_left", -beta)
            self.urdf_model.set_joint_angle("beta1_fake_rear_left", beta)
            self.urdf_model.set_joint_angle("beta2_fake_rear_left", -beta)
            self.urdf_model.set_joint_angle("beta_rear_right", -beta)
            self.urdf_model.set_joint_angle("beta1_fake_rear_right", beta)
            self.urdf_model.set_joint_angle("beta2_fake_rear_right", -beta)
            gamma = np.deg2rad(34.84)
            self.urdf_model.set_joint_angle("gamma_front_left", gamma)
            self.urdf_model.set_joint_angle("gamma1_fake_front_left", -gamma)
            self.urdf_model.set_joint_angle("gamma2_fake_front_left", gamma)
            self.urdf_model.set_joint_angle("gamma_front_right", gamma)
            self.urdf_model.set_joint_angle("gamma1_fake_front_right", -gamma)
            self.urdf_model.set_joint_angle("gamma2_fake_front_right", gamma)
            self.urdf_model.set_joint_angle("gamma_rear_left", gamma)
            self.urdf_model.set_joint_angle("gamma1_fake_rear_left", -gamma)
            self.urdf_model.set_joint_angle("gamma2_fake_rear_left", gamma)
            self.urdf_model.set_joint_angle("gamma_rear_right", gamma)
            self.urdf_model.set_joint_angle("gamma1_fake_rear_right", -gamma)
            self.urdf_model.set_joint_angle("gamma2_fake_rear_right", gamma)

        # Frames
        # ------
        # origin - origin point of the ground truth DEM in north, west, up

        # odometry - base frame of odometry, this frame is aligned with the body
        # frame of the robot at the initial position
        graph.add_frame("odometry")

        # center, is the starting position of the robot aligned to north, west, up

        # dgps0 - initial GPS position, this frame is aligned to the GPS frame as placed in the robot
        graph.add_frame("dgps0")

        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.array([327.3 - 0.3, 272.72, 0]))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
        graph.add_transform("center", "origin", t)

        # body - odometry, variable, updated continuously
        # origin - odometry, constant, known before start
        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.zeros(3))
        t.transform.orientation.fromarray(
            np.array([0.0, 0.0, -0.25881904510252074, 0.96592582628906831]))
        graph.add_transform("odometry", "center", t)

        # center - dgps0, constant, known before start
        t = cdff_envire.Transform()
        # This translation is from the first dgps point and the center that is the
        # starting position of the robot aligned to north, west, up
        t.transform.translation.fromarray(np.array([0.3, 0.0, 0.53]))
        t.transform.orientation.fromarray(np.array([1.0, 0.0, 0.0, 0.0]))

        graph.add_transform("dgps0", "center", t)

        # dgps - body
        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.array([0.3, 0.0, 0.53]))
        t.transform.orientation.fromarray(np.array(
            [-9.65427150e-01, 2.60673010e-01, -1.59616184e-17,
             -5.91153634e-17]))
        graph.add_transform("dgps", "body", t)

        map_slices = [
            [(5920, 7000), (5500, 7420)], [(7000, 8080), (5500, 7420)],
            [(5920, 7000), (3580, 5500)], [(7000, 8080), (3580, 5500)]
        ]
        geo_tiff = io.GeoTiffMap(os.path.join(log_folder, mapfile))
        for i, slice in enumerate(map_slices):
            start_point, map = geo_tiff.slice(
                map_slice_rows=slice[0], map_slice_cols=slice[1])
            # origin - groundtruth map, constant, known before start
            t = cdff_envire.Transform()
            t.transform.orientation.fromarray(
                np.array([0.0, 0.0, -0.70710678118654746, 0.70710678118654757]))
            t.transform.translation.fromarray(
                np.array([start_point[0], -start_point[1], -867.0]))
            graph.add_transform("origin", "map%d" % i, t)

            map_item = envirevisualization.EnvireItem(map)
            map_item.add_to(graph, "map%d" % i)
            self.map_items.append(map_item)

    def groundTruth2dgps0Input(self, data):
        if not self.gps_initialized:
            self.start_pos[:] = data.pos.toarray()
            self.start_orient.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))

            t = cdff_types.RigidBodyState()
            t.pos.fromarray(self.start_pos)
            t.orient = self.start_orient
            t.source_frame = "dgps0"
            t.target_frame = "start"
            t.timestamp.microseconds = self._timestamp
            self._set_transform(t, frame_transformation=False)

            self.gps_initialized = True

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

    def sherpaJointsInput(self, data):
        if not SHOW_JOINT_ANGLES:
            return

        for i in range(data.names.size()):
            name = data.names[i]
            if name.startswith("arm_joint"):
                continue
            angle = data.elements[i].position
            self.urdf_model.set_joint_angle(name, angle)

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
            self.odometry_timestamps_.append(
                odometry_pose.timestamp.microseconds)
            self.odometry_positions_.append(odometry_pose.pos.toarray())

    def groundTruth2originInput(self, ground_truth_pose):
        if ground_truth_pose.timestamp.microseconds > 0:
            self.ground_truth_timestamps_.append(
                ground_truth_pose.timestamp.microseconds)
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
        ("/mcs_sensor_processing.rigid_body_state_out",
         "transformer.body2odometry"),
        (
        "/mcs_sensor_processing.joints_status_all", "transformer.sherpaJoints"),

        # inputs to evaluation
        ("transformer.dgps2origin", "evaluation.dgps2origin"),
        ("transformer.groundTruth2origin", "evaluation.groundTruth2origin"),

        # inputs to trajectory
        ("transformer.groundTruth2origin", "trajectory_gt.pose"),
        ("transformer.dgps2origin", "trajectory_dgps.pose"),

        # outputs, this is necessary so register the output port in the dfc
        ("evaluation.error", "result.error"),
        ("trajectory_dgps.pos", "result.trajectory_dgps"),
        ("trajectory_gt.pos", "result.trajectory_gt"),
    )
    frames = {"trajectory_dgps.pos": "origin", "trajectory_gt.pos": "origin"}

    logfiles = [
        (odometry_logfile, ["/mcs_sensor_processing.rigid_body_state_out"]),
        (dgps_logfile, ["/dgps.imu_pose"])
    ]
    if SHOW_JOINT_ANGLES:
        logfiles += [
            (joints_logfile, ["/mcs_sensor_processing.joints_status_all"])]

    log_iterator = replay_logfile_join(log_folder, logfiles)

    app = envirevisualization.EnvireVisualizerApplication(
        frames, center_frame="center")
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
