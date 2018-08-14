import warnings
import numpy as np
from cdff_dev import logloader, envirevisualization, dataflowcontrol, transformer
import cdff_types
import cdff_envire
from cdff_dev.extensions.gps import conversion


# TODO configuration
dgps_logfile = "logs/Sherpa/dgps_Logger.log"
mcs_logfile = "logs/Sherpa/sherpa_tt_mcs_Logger.log"
utm_zone = 32  # TODO Utah 12, Germany 32, Morocco 29?
utm_north = True


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()
        self.imu_initialized = False

    def dgps2globalPose0Input(self, data):
        self._set_transform(data, data_transformation=True)

    """TODO extract orientation from IMU?
    def world2GlobalPoseInput(self, data):
        data.pos.fromarray(np.zeros(3))
        self._set_transform(data, data_transformation=True)
        if not self.imu_initialized:
            t = self.graph_.get_transform("world", "global_pose")
            self.graph_.add_transform("world0", "global_pose0", t)
            self.imu_initialized = True
    """

    def dgps2globalPose0Output(self):
        if (self.graph_.contains_frame("dgps") and
                self.graph_.contains_frame("global_pose0")):
            return self._get_transform(
                "dgps", "global_pose0", data_transformation=True)
        else:
            none = cdff_types.RigidBodyState()
            none.pos.fromarray(np.zeros(3))
            none.orient.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
            none.timestamp.microseconds = 0
            none.source_frame = "dgps"
            none.target_frame = "global_pose0"
            return none

    def body2odometryInput(self, data):
        # TODO works if False, but is actually True...
        self._set_transform(data, data_transformation=True)

    def body2odometryOutput(self):
        return self._get_transform("body", "odometry", data_transformation=True)

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
            self.relative_pose.source_frame = "dgps"
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

    def dgps2globalPose0Output(self):
        return self.relative_pose

    def _to_coordinates(self, gps, initial_pos=(0.0, 0.0, 0.0)):
        """Transform GPS coordinates to x, y, z coordinates."""
        easting, northing, altitude = conversion.convert_to_utm(
            gps.latitude, gps.longitude, gps.altitude,
            utm_zone=utm_zone, utm_north=utm_north)
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

    def dgps2globalPose0Input(self, gps_pose):
        if gps_pose.timestamp.microseconds > 0:
            self.dgps_timestamps_.append(gps_pose.timestamp.microseconds)
            self.dgps_positions_.append(gps_pose.pos.toarray())

    def body2odometryInput(self, odometry_pose):
        if odometry_pose.timestamp.microseconds > 0:
            self.odometry_timestamps_.append(odometry_pose.timestamp.microseconds)
            self.odometry_positions_.append(odometry_pose.pos.toarray())

    """ TODO
    def errorOutput(self):
        return self.error
    """

    def process(self):
        if not self.odometry_positions_ or not self.dgps_positions_:
            return
        self.error = np.linalg.norm(self.odometry_positions_[-1] -
                                    self.dgps_positions_[-1])
        print("[Evaluation] position error: %.2f" % self.error)


def main():
    logs, stream_names = convert_logs()
    app, dfc = configure(logs, stream_names)
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
    # TODO find mapping between global and local coordinate frames
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
        #("/dgps.imu_pose", "transformer.world2GlobalPose"),  # TODO extract orientation from IMU
        ("/dgps.gps_solution", "gps_to_relative_pose.gps"),
        ("gps_to_relative_pose.dgps2globalPose0", "transformer.dgps2globalPose0"),
        ("/mcs_sensor_processing.rigid_body_state_out", "transformer.body2odometry"),
        ("transformer.dgps2globalPose0", "evaluation.dgps2globalPose0"),
        ("transformer.body2odometry", "evaluation.body2odometry"),
    )
    frames = {}  # TODO?
    urdf_files = [
        # git clone git@git.hb.dfki.de:facilitators/bundle-sherpa_tt.git
        #"bundle-sherpa_tt/data/sherpa_tt.urdf"  # TODO uncomment
    ]

    log_iterator = logloader.replay_files(
        logs, stream_names, verbose=0)

    app = envirevisualization.EnvireVisualizerApplication(
        frames, urdf_files, center_frame="global_pose0")
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=False)
    dfc.setup()

    # TODO simplify graph initialization
    app.visualization.world_state_.graph_.add_frame("odometry")
    t = cdff_envire.Transform()
    t.transform.translation.fromarray(np.array([-0.3, 0.0, 0.53]))
    t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
    app.visualization.world_state_.graph_.add_transform(
        "odometry", "global_pose0", t)

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

    """
    dgps_t  = dfc.nodes["evaluation"].dgps_timestamps_
    odometry_t  = dfc.nodes["evaluation"].odometry_timestamps_

    plt.figure()
    plt.plot(dgps_t, [p[0] for p in dgps], label="DGPS")
    plt.plot(odometry_t, [p[0] for p in odometry], label="Odometry")
    plt.legend()

    plt.figure()
    plt.plot(dgps_t, [p[1] for p in dgps], label="DGPS")
    plt.plot(odometry_t, [p[1] for p in odometry], label="Odometry")
    plt.legend()

    plt.plot()
    """

    plt.show()
    # TODO implement
    #raise NotImplementedError()


if __name__ == "__main__":
    main()