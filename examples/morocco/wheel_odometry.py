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
        self._set_transform(data)

    def wheelOdometryInput(self, data):
        self._set_transform(data)


class GpsToRelativePoseDFN:
    def __init__(self):
        self.gps = cdff_types.GpsSolution()
        self.relative_pose = cdff_types.RigidBodyState()

        self.initial_pose = cdff_types.RigidBodyState()
        self.initial_pose.pos.fromarray(np.array([0.0, 0.0, 0.0]))
        self.initial_pose.orient.fromarray(np.array([1.0, 0.0, 0.0, 0.0]))
        self.initial_pose.source_frame = "world"
        self.initial_pose.target_frame = "start"
        self.initial_pose_set = False

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def gpsInput(self, gps):
        self.gps = gps

    def process(self):
        try:
            if not self.initial_pose_set:
                self.initial_pose.pos.fromarray(
                    np.array(self._to_coordinates(self.gps)))
                self.initial_pose.timestamp.microseconds = self.gps.time.microseconds
                self.initial_pose_set = True

            self.relative_pose.timestamp.microseconds = self.gps.time.microseconds
            self.relative_pose.pos.fromarray(
                np.array(self._to_coordinates(self.gps, self.initial_pose.pos)))
            self.relative_pose.source_frame = "start"
            self.relative_pose.target_frame = "dgps"
            self.relative_pose.orient.fromarray(np.array([1.0, 0.0, 0.0, 0.0]))
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

    def _to_coordinates(self, gps, initial_pose=(0.0, 0.0, 0.0)):
        """Transform GPS coordinates to x, y, z coordinates."""
        easting, northing, altitude = conversion.convert_to_utm(
            gps.latitude, gps.longitude, gps.altitude,
            utm_zone=utm_zone, utm_north=utm_north)
        northing, westing, up = conversion.convert_to_nwu(
            easting, northing, altitude,
            initial_pose[0], initial_pose[1], initial_pose[2])
        return northing, westing, up


class EvaluationDFN:
    def __init__(self):
        self.odometry_poses = []
        self.gps_poses = []
        self.error = 0.0

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def odometryPoseInput(self, odometry_pose):
        self.odometry_poses.append(odometry_pose)

    def gpsPoseInput(self, gps_pose):
        self.gps_poses.append(gps_pose)

    def process(self):
        if not self.odometry_poses or not self.gps_poses:
            return
        self.error = np.linalg.norm(self.odometry_poses[-1].pos.toarray() -
                                    self.gps_poses[-1].pos.toarray())
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
        ("gps_to_relative_pose.relativePose", "evaluation.gpsPose"),
        ("/mcs_sensor_processing.rigid_body_state_out", "evaluation.odometryPose"),
    )
    pose_frame = "start"  # TODO
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
        frames, urdf_files, center_frame="start")
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=False)
    dfc.setup()

    # TODO simplify graph initialization
    app.visualization.world_state_.graph_.add_frame("body")
    app.visualization.world_state_.graph_.add_frame("odometry")
    connect_gps_odometry = cdff_envire.Transform()
    connect_gps_odometry.transform.translation.fromarray(np.array([0.0, 0.0, 0.0]))
    connect_gps_odometry.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
    app.visualization.world_state_.graph_.add_transform(
        "start", "odometry", connect_gps_odometry)

    # TODO visualize covariance?

    app.show_controls(log_iterator, dfc)

    return app, dfc


def evaluate(dfc):
    # TODO implement
    pass
    #raise NotImplementedError()


if __name__ == "__main__":
    main()