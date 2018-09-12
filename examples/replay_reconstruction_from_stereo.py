import glob
import numpy as np
from cdff_dev import dataflowcontrol, logloader, path, envirevisualization, cvvisualizer, replay
# TODO remove clone from path
from cdff_dev.dfpcs.reconstruction3d.reconstruction3d import EstimationFromStereo
import cdff_envire
from PyQt4.QtGui import QApplication


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
    trigger_ports = {
        "reconstruction3d": "rightImage"
    }
    connections = (
        ("/hcru0/pt_stereo_rect/left.image", "reconstruction3d.leftImage"),
        ("/hcru0/pt_stereo_rect/right.image", "reconstruction3d.rightImage"),

        #("/hcru0/pt_stereo_rect/left.image", "result.leftImage"),
        #("/hcru0/pt_stereo_rect/right.image", "result.rightImage"),

        ("reconstruction3d.pointCloud", "result.pointCloud"),
        ("reconstruction3d.pose", "result.pose"),
        ("reconstruction3d.success", "result.success"),
    )

    stream_names = [
        "/hcru0/pt_stereo_rect/left/image",
        "/hcru0/pt_stereo_rect/right/image",
    ]
    stream_aliases = {
        "/hcru0/pt_stereo_rect/left/image": "/hcru0/pt_stereo_rect/left.image",
        "/hcru0/pt_stereo_rect/right/image": "/hcru0/pt_stereo_rect/right.image",
    }
    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    log_folder = "logs/DLR_20180724/"
    logfiles = [
        #[log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_left_image_%09d.msg" % i for i in range(2)],
        sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_left_image_*.msg")),
        #[log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_right_image_%09d.msg" % i for i in range(2)],
        sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_right_image_*.msg")),
    ]
    log_iterator = logloader.replay_files(logfiles, stream_names)

    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, trigger_ports=trigger_ports,
        stream_aliases=stream_aliases, verbose=verbose)
    dfc.setup()

    vis = cvvisualizer.CVVisualizer("/hcru0/pt_stereo_rect/left.image")
    dfc.set_visualization(vis)

    replay.replay_and_process(dfc, log_iterator)

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
