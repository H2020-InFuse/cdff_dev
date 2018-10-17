import os
from cdff_dev import dataflowcontrol, logloader, imagevisualization, dfnhelpers


def main():
    verbose = 2

    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    log_folder = "logs/sherpa_hcru"
    prefix = "recording_20180927-175146_sherpaTT_integration"

    prefix_path = os.path.join(log_folder, prefix)

    merge_frame_pair = dfnhelpers.MergeFramePairDFN(
        left_camera_info_stream="/hcru1/pt_stereo_rect/left/camera_info",
        right_camera_info_stream="/hcru1/pt_stereo_rect/right/camera_info",
        left_is_main_camera=True, verbose=1
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

    # TODO show frames in envire visualizer

    log_iterator = logloader.replay_logfile_sequence(
        logloader.group_pattern(prefix_path, "_0*.msg"),
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
