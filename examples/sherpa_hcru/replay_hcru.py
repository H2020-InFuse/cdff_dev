import glob
from cdff_dev import dataflowcontrol, logloader, visualization_qtgraph


def main():
    verbose = 2
    connections = (
        ("/hcru1/pt_stereo_rect/left.image", "result.leftImage"),
        ("/hcru1/pt_stereo_rect/right.image", "result.rightImage"),
        ("/hcru1/pt_color/left.image", "result.rgbImage"),
        ("/hcru1/pt_stereo_sgm.depth", "result.depthImage"),
    )
    stream_names = [
        "/hcru1/pt_stereo_rect/left/image",
        "/hcru1/pt_stereo_rect/right/image",
        "/hcru1/pt_color/left/image",
        "/hcru1/pt_stereo_sgm/depth",
    ]
    stream_aliases = {
        "/hcru1/pt_stereo_rect/left/image": "/hcru1/pt_stereo_rect/left.image",
        "/hcru1/pt_stereo_rect/right/image": "/hcru1/pt_stereo_rect/right.image",
        "/hcru1/pt_color/left/image": "/hcru1/pt_color/left.image",
        "/hcru1/pt_stereo_sgm/depth": "/hcru1/pt_stereo_sgm.depth",
    }
    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    #log_folder = "/home/dfki.uni-bremen.de/afabisch/logs/20180926_sherpa_hcru/"
    log_folder = "/media/afabisch/TOSHIBA EXT/Datasets/20180927_sherpa_and_hcru_log/converted/"
    #prefix = "recording_20180927-175146_sherpaTT_integration"
    prefix = "recording_20180927-175540_sherpaTT_integration"
    logfiles = [
        #sorted(glob.glob(log_folder + prefix + "_hcru1_pt_stereo_rect_left_image_*.msg")),
        #sorted(glob.glob(log_folder + prefix + "_hcru1_pt_stereo_rect_right_image_*.msg")),
        sorted(glob.glob(log_folder + prefix + "_hcru1_pt_color_left_image_*.msg")),
        #sorted(glob.glob(log_folder + prefix + "_hcru1_pt_stereo_sgm_depth_*.msg")),
    ]
    if len(logfiles[0]) == 0:
        exit()
    log_iterator = logloader.replay_files(logfiles, stream_names)

    dfc = dataflowcontrol.DataFlowControl(
        nodes={}, connections=connections, stream_aliases=stream_aliases,
        verbose=verbose)
    dfc.setup()

    app = visualization_qtgraph.ImageVisualizerApplication(
        "/hcru1/pt_color/left.image")
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
