import os
import msgpack
import numpy as np
from cdff_dev import dataflowcontrol, logloader, imagevisualization
import cdff_types


class MergeFramePairDFN:
    def __init__(self, left_camera_info_stream, right_camera_info_stream):
        self.left_camera_info_stream = left_camera_info_stream
        self.right_camera_info_stream = right_camera_info_stream
        self.config_filename = None
        self.left_config = {}
        self.right_config = {}
        self.left_image = None
        self.right_image = None
        self.pair = cdff_types.FramePair()

    def set_configuration_file(self, filename):
        self.config_filename = filename

    def configure(self):
        with open(self.config_filename, "rb") as f:
            self.camera_configs = msgpack.unpack(
                f, encoding="utf8", use_list=False)
            self.left_config = self.camera_configs[
                self.left_camera_info_stream]
            self.right_config = self.camera_configs[
                self.right_camera_info_stream]
            self.pair.baseline = self.right_config["baseline"]

            ###### TODO remove
            print(self.left_config)
            print(self.right_config)
            ######

    def leftImageInput(self, data):
        self.left_image = data

    def rightImageInput(self, data):
        self.right_image = data

    def process(self):
        self.pair.left = self.left_image
        self.pair.right = self.right_image

        self.pair.left.intrinsic.dist_coeffs.fromarray(
            np.array(self.left_config["intrinsic"]["distCoeffs"]))
        self.pair.left.intrinsic.camera_matrix.fromarray(
            np.array(self.left_config["intrinsic"]["cameraMatrix"]))
        self.pair.left.intrinsic.camera_model = self.left_config[
            "intrinsic"]["cameraModel"]

        self.pair.right.intrinsic.dist_coeffs.fromarray(
            np.array(self.right_config["intrinsic"]["distCoeffs"]))
        self.pair.right.intrinsic.camera_matrix.fromarray(
            np.array(self.right_config["intrinsic"]["cameraMatrix"]))
        self.pair.right.intrinsic.camera_model = self.right_config[
            "intrinsic"]["cameraModel"]

        ###### TODO remove
        print(self.pair)
        ######

    def pairOutput(self):
        return self.pair


def main():
    verbose = 2

    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    log_folder = "logs/sherpa_hcru"
    prefix = "recording_20180927-175146_sherpaTT_integration"

    prefix_path = os.path.join(log_folder, prefix)

    merge_frame_pair = MergeFramePairDFN(
        left_camera_info_stream="/hcru1/pt_stereo_rect/left/camera_info",
        right_camera_info_stream="/hcru1/pt_stereo_rect/right/camera_info"
    )
    merge_frame_pair.set_configuration_file(prefix_path + "_camera.msg")

    nodes = {
        "merge_frame_pair": merge_frame_pair
    }
    trigger_ports = {
        "merge_frame_pair": "rightImage"
    }
    connections = (
        ("/hcru1/pt_stereo_rect/left.image", "merge_frame_pair.leftImage"),
        ("/hcru1/pt_stereo_rect/right.image", "merge_frame_pair.rightImage"),
        ("/hcru1/pt_color/left.image", "result.rgbImage"),
        ("/hcru1/pt_stereo_sgm.depth", "result.depthImage"),
        ("merge_frame_pair.pair", "result.pair"),
    )
    stream_aliases = {
        "/hcru1/pt_stereo_rect/left/image": "/hcru1/pt_stereo_rect/left.image",
        "/hcru1/pt_stereo_rect/right/image": "/hcru1/pt_stereo_rect/right.image",
        "/hcru1/pt_color/left/image": "/hcru1/pt_color/left.image",
        "/hcru1/pt_stereo_sgm/depth": "/hcru1/pt_stereo_sgm.depth",
    }

    log_iterator = logloader.replay_join([
        logloader.replay_logfile(
            filename,
            ["/hcru1/pt_stereo_rect/left/image",
             "/hcru1/pt_stereo_rect/right/image",
             "/hcru1/pt_color/left/image",
             "/hcru1/pt_stereo_sgm/depth"]
        )
        for filename in logloader.group_pattern(prefix_path, "_0*.msg")
    ])

    dfc = dataflowcontrol.DataFlowControl(
        nodes=nodes, connections=connections, trigger_ports=trigger_ports,
        stream_aliases=stream_aliases, verbose=verbose)
    dfc.setup()

    #app = imagevisualization.ImageVisualizerApplication("/hcru1/pt_stereo_rect/left.image")
    #app = imagevisualization.ImageVisualizerApplication("/hcru1/pt_stereo_rect/right.image")
    #app = imagevisualization.ImageVisualizerApplication("/hcru1/pt_color/left.image")
    #app = imagevisualization.ImageVisualizerApplication("/hcru1/pt_stereo_sgm.depth", (0.0, 3.0))
    app = imagevisualization.ImagePairVisualizerApplication("merge_frame_pair.pair")
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
