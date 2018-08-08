import numpy as np
from cdff_dev import logloader, envirevisualization, dataflowcontrol, transformer
import cdff_types
from cdff_dev.extensions.gps import conversion


# TODO configuration
dgps_logfile = "logs/Sherpa/dgps_Logger.log"
mcs_logfile = "logs/Sherpa/sherpa_tt_mcs_Logger.log"
utm_zone = 12  # TODO Utah 12, Germany 32, Morocco 29?
utm_north = True


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()

    def relativePoseInput(self, data):
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
        if not self.initial_pose_set:
            self.initial_pose.pos.fromarray(
                np.array(self._to_coordinates(gps)))
            self.initial_pose.timestamp.microseconds = gps.time.microseconds
            self.initial_pose_set = True

    def process(self):
        self.relative_pose.timestamp.microseconds = self.gps.time.microseconds
        self.relative_pose.pos.fromarray(
            np.array(self._to_coordinates(self.gps, self.initial_pose.pos)))
        self.relative_pose.source_frame = "start"
        self.relative_pose.target_frame = "dgps"
        self.relative_pose.orient.fromarray(np.array([1.0, 0.0, 0.0, 0.0]))

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
        self.odometry_pose = cdff_types.RigidBodyState()
        self.gps_pose = cdff_types.RigidBodyState()
        self.error = 0.0

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def odometryPoseInput(self, odometry_pose):
        self.odometry_pose = odometry_pose


    def gpsPoseInput(self, gps_pose):
        self.gps_pose = gps_pose

    def process(self):
        raise NotImplementedError()

    def errorOutput(self):
        return self.error


def main():
    logs = convert_logs()
    app, dfc = configure(logs)
    app.exec_()
    evaluate(dfc)


def convert_logs():
    # TODO implement
    return "logs/dgps.msg"
    #raise NotImplementedError()


def configure(logs):
    nodes = {
        "gps_to_relative_pose": GpsToRelativePoseDFN(),
        "transformer": Transformer(),
        #"evaluation": EvaluationDFN()
    }
    periods = {
        "gps_to_relative_pose": 0.1,  # TODO
        "transformer": 0.1 # TODO
        #"evaluation": 0.1 # TODO
    }
    connections = (
        ("/dgps.gps_solution", "gps_to_relative_pose.gps"),
        ("gps_to_relative_pose.relativePose", "transformer.relativePose"),
        #("gps_to_relative_pose.relativePose", "evaluation.gpsPose"),
        #("/?.?", "evaluation.odometryPose"),  # TODO
    )
    pose_frame = "start"  # TODO
    frames = {
        "gps_to_relative_pose.gps": pose_frame,  # TODO
        #"?.?": pose_frame,  # TODO
    }
    urdf_files = [
        #"sherpa?.urdf"  # TODO
    ]

    log = logloader.load_log(  # TODO use multiple files
        logs
    )
    stream_names = [
        "/dgps.gps_solution",
        #"/?.?"  # TODO
    ]
    log_iterator = logloader.replay(stream_names, log)

    app = envirevisualization.EnvireVisualizerApplication(
        frames, urdf_files, center_frame="start")
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=False)
    dfc.setup()

    app.show_controls(log_iterator, dfc)

    return app, dfc


def evaluate(dfc):
    # TODO implement
    pass
    #raise NotImplementedError()


if __name__ == "__main__":
    main()