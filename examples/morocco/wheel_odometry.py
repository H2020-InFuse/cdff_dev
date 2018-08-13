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

    def relativePoseInput(self, data):
        self._set_transform(data, data_transformation=True)

    def wheelOdometryInput(self, data):
        self._set_transform(data, data_transformation=False)  # TODO works if False, but is actually True...

    def process(self):
        super(Transformer, self).process()

    def odometry2bodyOutput(self):
        return self._get_transform("odometry", "body", data_transformation=True)

    def odometry2dgpsOutput(self):
        if self.graph_.contains_frame("dgps"):
            return self._get_transform("odometry", "dgps", data_transformation=True)
        else:
            none = cdff_types.RigidBodyState()
            none.pos.fromarray(np.zeros(3))
            none.orient.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
            none.timestamp.microseconds = 0
            none.source_frame = "odometry"
            none.target_frame = "dgps"
            return none


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
            self.relative_pose.target_frame = "odometry"
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

    def relativePoseOutput(self):
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

    def odometry2bodyInput(self, odometry_pose):
        if odometry_pose.timestamp.microseconds > 0:
            self.odometry_timestamps_.append(odometry_pose.timestamp.microseconds)
            self.odometry_positions_.append(odometry_pose.pos.toarray())

    def odometry2dgpsInput(self, gps_pose):
        if gps_pose.timestamp.microseconds > 0:
            self.dgps_timestamps_.append(gps_pose.timestamp.microseconds)
            self.dgps_positions_.append(gps_pose.pos.toarray())

    def process(self):
        if not self.odometry_positions_ or not self.dgps_positions_:
            return
        self.error = np.linalg.norm(self.odometry_positions_[-1] -
                                    self.dgps_positions_[-1])
        print("[Evaluation] position error: %.2f" % self.error)

    def errorOutput(self):
        return self.error


def main():
    logs = convert_logs()
    app, dfc = configure(logs)
    app.exec_()
    evaluate(dfc)


def convert_logs():
    # TODO implement
    # TODO find number of samples per stream
    # TODO chunk logfiles
    return [["logs/Sherpa/dgps.msg"], ["logs/Sherpa/sherpa_tt_mcs.msg"]]
    #raise NotImplementedError()


def configure(logs):
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
        ("/dgps.gps_solution", "gps_to_relative_pose.gps"),
        ("gps_to_relative_pose.relativePose", "transformer.relativePose"),
        ("/mcs_sensor_processing.rigid_body_state_out", "transformer.wheelOdometry"),
        ("transformer.odometry2dgps", "evaluation.odometry2dgps"),
        ("transformer.odometry2body", "evaluation.odometry2body"),
    )
    pose_frame = "odometry"  # TODO
    frames = {
        "gps_to_relative_pose.gps": pose_frame,  # TODO
        #"?.?": pose_frame,  # TODO
    }
    urdf_files = [
        #"sherpa?.urdf"  # TODO
    ]

    log_iterator = logloader.replay_files(
        logs,
        stream_names = [
            "/dgps.gps_solution",
            "/mcs_sensor_processing.rigid_body_state_out"
        ],
        verbose=0
    )

    app = envirevisualization.EnvireVisualizerApplication(
        frames, urdf_files, center_frame="odometry")
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=False)
    dfc.setup()

    # TODO simplify graph initialization

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