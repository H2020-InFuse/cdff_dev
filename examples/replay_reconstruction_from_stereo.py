import glob
from cdff_dev import dataflowcontrol, visualization2d, path
# TODO remove clone from path
from cdff_dev.dfpcs.reconstruction3d.reconstruction3d import EstimationFromStereo


class DfpcAsDfn:  # TODO make this a general solution or find a better one
    def __init__(self, dfpc):
        self.dfpc = dfpc

    def configure(self):
        self.dfpc.setup()

    def process(self):
        self.dfpc.run()

    def leftImageInput(self, data):
        self.dfpc.leftImageInput(data)

    def rightImageInput(self, data):
        self.dfpc.rightImageInput(data)

    def pointCloudOutput(self):
        data = self.dfpc.pointCloudOutput()
        return data

    def poseOutput(self):
        data = self.dfpc.poseOutput()
        return data

    def successOutput(self):
        data = self.dfpc.successOutput()
        return data

    """ # TODO better meta programming... create class on the fly
    def __getattr__(self, name):
        if name.endswith("Input") or name.endswith("Output"):
            return self.dfpc.__getattr__(name)
        elif name == "configure":
            return self.dfpc.setup
        elif name == "process":
            return self.dfpc.run
        else:
            raise AttributeError("No attribute with name '%s'" % name)
    """


def main():
    verbose = 0
    reconstruction3d = EstimationFromStereo()
    # TODO install configuration files?
    config_filename = path.load_cdffpath() + "/Tests/ConfigurationFiles/DFPCs/Reconstruction3D/DfpcEstimationFromStereo_DlrHcru.yaml"
    reconstruction3d.set_configuration_file(config_filename)
    nodes = {
        "reconstruction3d": DfpcAsDfn(reconstruction3d)
    }
    periods = {
        #"reconstruction3d": 1.0  # TODO
    }
    trigger_ports = {
        "reconstruction3d": "rightImage"
    }
    connections = (
        #("/hcru0.pose_cov", "result.pose"),
        ("/hcru0/pt_stereo_rect/left.image", "reconstruction3d.leftImage"),
        ("/hcru0/pt_stereo_rect/right.image", "reconstruction3d.rightImage"),

        ("reconstruction3d.pointCloud", "result.pointCloud"),
        ("reconstruction3d.pose", "result.pose"),
        ("reconstruction3d.success", "result.success"),
    )

    stream_names = [
        #"/hcru0/pose_cov",
        "/hcru0/pt_stereo_rect/left/image",
        "/hcru0/pt_stereo_rect/right/image",
    ]
    stream_aliases = {
        #"/hcru0/pose_cov": "/hcru0.pose_cov",
        "/hcru0/pt_stereo_rect/left/image": "/hcru0/pt_stereo_rect/left.image",
        "/hcru0/pt_stereo_rect/right/image": "/hcru0/pt_stereo_rect/right.image",
    }
    image_stream_names = [
        "/hcru0/pt_stereo_rect/left.image",
        "/hcru0/pt_stereo_rect/right.image"
    ]
    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    log_folder = "logs/DLR_20180724/"
    logfiles = [
        #[log_folder + "recording_20180724-135036_hcru0_pose_cov_%09d.msg" % i for i in range(5)],
        #sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pose_cov_*.msg")),
        [log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_left_image_%09d.msg" % i for i in range(2)],
        #sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_left_image_*.msg")),
        [log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_right_image_%09d.msg" % i for i in range(2)],
        #sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_right_image_*.msg"))
    ]

    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods=periods, trigger_ports=trigger_ports,
        stream_aliases=stream_aliases, verbose=verbose)
    #dfc.setup()  # TODO why is this done already?

    vis = visualization2d.MatplotlibVisualizerApplication()
    vis.show_controls(
        dfc, logfiles, stream_names, image_stream_names=image_stream_names,
        image_shape=(1032, 772))
    vis.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
