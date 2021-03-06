"""
========
HCRU Log
========

In this example we will replay camera images and display a stereo image pair.

The log data is available from Zenodo at

    TODO
"""
import os
from cdff_dev import dataflowcontrol, logloader, imagevisualization, dfnhelpers


def main():
    verbose = 0

    log_folder = "logs/open_hcru_bremen"
    prefix = "recording_20180927-175540_sherpaTT_integration"

    prefix_path = os.path.join(log_folder, prefix)

    merge_frame_pair = dfnhelpers.MergeFramePairDFN(
        left_camera_info_stream="/hcru1/pt_stereo_rect/left/camera_info",
        right_camera_info_stream="/hcru1/pt_stereo_rect/right/camera_info",
        left_is_main_camera=True, verbose=verbose
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
    )
    stream_aliases = {
        "/hcru1/pt_stereo_rect/left/image": "/hcru1/pt_stereo_rect/left.image",
        "/hcru1/pt_stereo_rect/right/image": "/hcru1/pt_stereo_rect/right.image",
        "/hcru1/pt_color/left/image": "/hcru1/pt_color/left.image",
        "/hcru1/pt_stereo_sgm/depth": "/hcru1/pt_stereo_sgm.depth",
    }

    log_iterator = logloader.replay_logfile_sequence(
        logloader.group_pattern(prefix_path, "*.msg"),
        ["/hcru1/pt_stereo_rect/left/image",
         "/hcru1/pt_stereo_rect/right/image",
         "/hcru1/pt_color/left/image",
         "/hcru1/pt_stereo_sgm/depth"])

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
