from cdff_dev import logloader, envirevisualization, dataflowcontrol
import cdff_types


class GpsToRelativePoseDFN:
    def __init__(self):
        self.gps = cdff_types.GpsSolution()
        self.relative_pose = cdff_types.RigidBodyState()

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def gpsInput(self, gps):
        self.gps = gps

    def process(self):
        raise NotImplementedError()

    def relativePoseOutput(self):
        return self.relative_pose


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
    convert_logs()
    app, dfc = configure()
    app.exec_()
    evaluate(dfc)


def convert_logs():
    raise NotImplementedError()


def configure():
    nodes = {
        "gps_to_relative_pose": GpsToRelativePoseDFN(),
        "evaluation": EvaluationDFN()
    }
    periods = {
        "gps_to_relative_pose": 0.1,  # TODO
        "evaluation": 0.1 # TODO
    }
    connections = (
        ("/dgps.gps_solution", "gps_to_relative_pose.gps"),
        ("gps_to_relative_pose.relativePose", "evaluation.gpsPose"),
        ("/?.?", "evaluation.odometryPose"),  # TODO
    )
    pose_frame = "?"  # TODO
    frames = {
        "gps_to_relative_pose.gps": pose_frame,  # TODO
        "?.?": pose_frame,  # TODO
    }
    urdf_files = [
        "sherpa?.urdf"  # TODO
    ]

    log = logloader.load_log(  # TODO use multiple files
        "?"  # TODO
    )
    stream_names = [
        "/dgps.gps_solution",
        "/?.?"  # TODO
    ]

    app = envirevisualization.EnvireVisualizerApplication(frames, urdf_files)
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, real_time=True)
    dfc.setup()

    app.show_controls(stream_names, log, dfc)

    return app, dfc


def evaluate(dfc):
    raise NotImplementedError()


if __name__ == "__main__":
    main()