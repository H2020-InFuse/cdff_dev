import warnings
import numpy as np
from cdff_dev import logloader, envirevisualization, dataflowcontrol, transformer
import cdff_types
import cdff_envire
from cdff_dev.extensions.gps import conversion
# TODO dependency: pytransform; git@github.com:rock-learning/pytransform.git
import pytransform.rotations as pr


# TODO transformations:
# - convert rock transformations to source in target
# - invert pytransform rotations

# TODO configuration
dgps_logfile = "logs/Sherpa/dgps_Logger.log"
mcs_logfile = "logs/Sherpa/sherpa_tt_mcs_Logger.log"
utm_zone = 32  # TODO Utah 12, Germany 32, Morocco 29?
utm_north = True


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()
        self.imu_initialized = False

    def gpsPos2globalPose0Input(self, data):
        self._set_transform(data, frame_transformation=False)

    def imu02GlobalPose0Input(self, data):
        if not self.imu_initialized:
            t = cdff_types.RigidBodyState()
            t.pos.fromarray(np.zeros(3))
            t.orient.fromarray(data.orient.toarray())
            t.source_frame = "global_pose0"
            t.target_frame = "imu0"
            self._set_transform(t, frame_transformation=False)

            self.imu_initialized = True

    def gpsPos2odometryOutput(self):
        return self._get_transform(
            "gps_pos", "odometry", frame_transformation=False)

    def body2odometryInput(self, data):
        self._set_transform(data, frame_transformation=False)

    def process(self):
        super(Transformer, self).process()


class GpsToRelativePoseDFN:
    def __init__(self):
        self.gps = cdff_types.GpsSolution()
        self.relative_pose = cdff_types.RigidBodyState()

        self.initial_pos = (0.0, 0.0, 0.0)
        self.initial_pos_set = False
        self.input_provided = False

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def gpsInput(self, gps):
        self.gps = gps
        self.input_provided = True

    def process(self):
        if not self.input_provided:
            return
        try:
            if not self.initial_pos_set:
                self.initial_pos = self._to_coordinates(self.gps)
                self.initial_pos_set = True

            self.relative_pose.timestamp.microseconds = self.gps.time.microseconds
            self.relative_pose.pos.fromarray(
                np.array(self._to_coordinates(self.gps, self.initial_pos)))
            self.relative_pose.source_frame = "gps_pos"
            self.relative_pose.target_frame = "global_pose0"
            self.relative_pose.orient.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
            self.relative_pose.cov_position[0, 0] = \
                self.gps.deviation_latitude ** 2
            self.relative_pose.cov_position[1, 1] = \
                self.gps.deviation_longitude ** 2
            self.relative_pose.cov_position[2, 2] = \
                self.gps.deviation_altitude ** 2
        except Exception as e:
            warnings.warn("Failure with GPS coordinates latitude=%g, "
                          "longitude=%g, altitude=%g, error: %s"
                          % (self.gps.latitude, self.gps.longitude,
                             self.gps.altitude, e))

    def gpsPos2globalPose0Output(self):
        return self.relative_pose

    def _to_coordinates(self, gps, initial_pos=(0.0, 0.0, 0.0)):
        """Transform GPS coordinates to x, y, z coordinates."""
        easting, northing, altitude = conversion.convert_to_utm(
            gps.latitude, gps.longitude, gps.altitude,
            utm_zone=utm_zone, utm_north=utm_north)

        # NWU
        northing, westing, up = conversion.convert_to_nwu(
            easting, northing, altitude,
            initial_pos[0], initial_pos[1], initial_pos[2])
        return northing, westing, up


class EvaluationDFN:
    def __init__(self):
        self.odometry_timestamps_ = []
        self.odometry_positions_ = []
        self.dgps_timestamps_ = []
        self.dgps_positions_ = []
        self.error = 0.0

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def gpsPos2odometryInput(self, gps_pose):
        if gps_pose.timestamp.microseconds > 0:
            self.dgps_timestamps_.append(gps_pose.timestamp.microseconds)
            self.dgps_positions_.append(gps_pose.pos.toarray())

    def body2odometryInput(self, odometry_pose):
        if odometry_pose.timestamp.microseconds > 0:
            self.odometry_timestamps_.append(odometry_pose.timestamp.microseconds)
            self.odometry_positions_.append(odometry_pose.pos.toarray())

    def errorOutput(self):
        return self.error

    def process(self):
        if not self.odometry_positions_ or not self.dgps_positions_:
            return
        self.error = np.linalg.norm(self.odometry_positions_[-1] -
                                    self.dgps_positions_[-1])
        print("[Evaluation] position error: %.2f" % self.error)


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
    # - /dgps.gps_solution
    # - /dgps.imu_pose
    # logs/Sherpa/sherpa_tt_mcs.msg
    # - /mcs_sensor_processing.rigid_body_state_out

    logfiles = [["logs/Sherpa/dgps.msg"], ["logs/Sherpa/sherpa_tt_mcs.msg"]]
    stream_names = [
       "/dgps.gps_solution",
       "/dgps.imu_pose",
       "/mcs_sensor_processing.rigid_body_state_out",
    ]

    return logfiles, stream_names


def configure(logs, stream_names):
    nodes = {
        "gps_to_relative_pose": GpsToRelativePoseDFN(),
        "transformer": Transformer(),
        "evaluation": EvaluationDFN()
        # TODO conversion to path
    }
    periods = {
        # frequency of odometry: 0.01
        # frequency of gps: 0.05
        "gps_to_relative_pose": 0.05,  # TODO
        "transformer": 0.01,
        "evaluation": 1.0, # TODO
    }
    connections = (
        ("/dgps.imu_pose", "transformer.imu02GlobalPose0"),
        ("/dgps.gps_solution", "gps_to_relative_pose.gps"),
        ("gps_to_relative_pose.gpsPos2globalPose0", "transformer.gpsPos2globalPose0"),
        ("/mcs_sensor_processing.rigid_body_state_out", "transformer.body2odometry"),
        ("transformer.gpsPos2odometry", "evaluation.gpsPos2odometry"),
        ("/mcs_sensor_processing.rigid_body_state_out", "evaluation.body2odometry"),
        ("evaluation.error", "result.error"),
    )
    frames = {}  # TODO?
    urdf_files = [
        # git clone git@git.hb.dfki.de:facilitators/bundle-sherpa_tt.git
        #"bundle-sherpa_tt/data/sherpa_tt.urdf"  # TODO uncomment
    ]

    log_iterator = logloader.replay_files(logs, stream_names, verbose=0)

    app = envirevisualization.EnvireVisualizerApplication(
        frames, urdf_files, center_frame="body0")
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
    # body0 - initial position of the robot according to odometry, same as
    #         odometry
    #graph.add_frame("body0") - already in the graph (center frame)
    # imu0 - imu sensor frame on the robot, use to connect body0 to initial imu
    #        measurements
    graph.add_frame("imu0")
    ##graph.add_frame("dgps0")
    # global_pose0 - initial GPS position with axes aligned to north, west, up
    graph.add_frame("global_pose0")
    # gps_pos - current GPS position
    graph.add_frame("gps_pos")

    # body - odometry, variable, updated continuously
    # body0 - odometry, constant, known before start
    t = cdff_envire.Transform()
    t.transform.translation.fromarray(np.zeros(3))
    t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
    graph.add_transform("body0", "odometry", t)
    # body0 - imu0, constant, known before start
    t = cdff_envire.Transform()
    t.transform.translation.fromarray(np.array([-0.185, 0.3139, 0.04164]))
    R = pr.matrix_from_euler_xyz([0.44, 2.225, 0.0])
    imu2body_wxyz = pr.quaternion_from_matrix(R.T)
    imu2body_xyzw = np.hstack((imu2body_wxyz[1:], [imu2body_wxyz[0]]))
    t.transform.orientation.fromarray(imu2body_xyzw)
    graph.add_transform("imu0", "body0", t)
    # imu0 - global_pose0, constant, known after start
    # gps_pos - global_pose0, variable, updated continuously

    # TODO visualize covariance?

    app.show_controls(log_iterator, dfc)

    return app, dfc


def evaluate(dfc):
    dgps = dfc.nodes["evaluation"].dgps_positions_
    odometry = dfc.nodes["evaluation"].odometry_positions_
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(aspect="equal")
    plt.grid()
    plt.plot([p[0] for p in dgps], [p[1] for p in dgps], label="DGPS")
    plt.plot([p[0] for p in odometry], [p[1] for p in odometry], label="Odometry")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()