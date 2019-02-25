"""
=============================
Replay 3D Reconstruction DFPC
=============================

This script will replay log files, run the DFPC, and visualize results.
The dataset is available from Zenodo at

    https://zenodo.org/record/2576885

Copy them to the folder 'logs/open_morocco_reconstruction3d' to run
this script.
"""
print(__doc__)
import os
import numpy as np
from cdff_dev import (dataflowcontrol, logloader, path, transformer,
                      envirevisualization, imagevisualization)
from cdff_dev.dfpcs.reconstruction3d import DenseRegistrationFromStereo
import cdff_envire


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        transformer.EnvireDFN.__init__(self)

    def initialize_graph(self, graph):
        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.zeros(3))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
        graph.add_transform("config_sherpaTT_body", "origin", t)


def main():
    dfc = initialize_dfc(verbose=2)
    log_iterator = initialize_log_iterator()

    app = envirevisualization.EnvireVisualizerApplication(
        frames={"reconstruction3d.pointCloud": "config_camera_left"},
        center_frame="origin")

    visualization = imagevisualization.ImageVisualization(
        "/hcru1/pt_stereo_rect/left.image")
    dfc.register_visualization(visualization)

    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


def initialize_log_iterator():
    log_folder = "logs/open_morocco_reconstruction3d/"
    filenames = logloader.group_pattern(
        log_folder, "recording_20181202_0*.msg")
    stream_names = ["/hcru1/pt_stereo_rect/left/image",
                    "/hcru1/pt_stereo_rect/right/image"]
    log_iterator = logloader.replay_logfile_sequence(filenames, stream_names)
    return log_iterator


def initialize_dfc(verbose):
    reconstruction3d = DenseRegistrationFromStereo()
    config_filename = os.path.join(
        path.load_cdffpath(),
        "Tests/ConfigurationFiles/DFPCs/Reconstruction3D/"
        "DfpcDenseRegistrationFromStereo_DlrHcru.yaml")
    reconstruction3d.set_configuration_file(config_filename)
    transformer = Transformer()
    transformer.set_configuration_file(
        "logs/open_morocco_reconstruction3d/recording_20181202_tf.msg")
    nodes = {"reconstruction3d": reconstruction3d, "transformer": transformer}
    connections = (
        ("/hcru1/pt_stereo_rect/left.image", "reconstruction3d.leftImage"),
        ("/hcru1/pt_stereo_rect/right.image", "reconstruction3d.rightImage"),

        ("reconstruction3d.pointCloud", "result.pointCloud"),
        ("reconstruction3d.pose", "result.pose"),
        ("reconstruction3d.success", "result.success"),
    )
    stream_aliases = {
        "/hcru1/pt_stereo_rect/left/image": "/hcru1/pt_stereo_rect/left.image",
        "/hcru1/pt_stereo_rect/right/image": "/hcru1/pt_stereo_rect/right.image",
    }
    periods = {"transformer": 1.0, "reconstruction3d": 1.0}
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods=periods, stream_aliases=stream_aliases,
        verbose=verbose)
    dfc.setup()
    return dfc


if __name__ == "__main__":
    main()
