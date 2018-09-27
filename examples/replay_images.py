import glob
from cdff_dev import dataflowcontrol, logloader, visualization_qtgraph


def main():
    verbose = 2
    connections = (
        ("/hcru0/pt_stereo_rect/left.image", "result.leftImage"),
        ("/hcru0/pt_stereo_rect/right.image", "result.rightImage"),
        ("/hcru0/pt_stereo_sgm.depth", "result.depthImage")
    )
    stream_names = [
        "/hcru0/pt_stereo_rect/left/image",
        "/hcru0/pt_stereo_rect/right/image",
        "/hcru0/pt_stereo_sgm/depth",
    ]
    stream_aliases = {
        "/hcru0/pt_stereo_rect/left/image": "/hcru0/pt_stereo_rect/left.image",
        "/hcru0/pt_stereo_rect/right/image": "/hcru0/pt_stereo_rect/right.image",
        "/hcru0/pt_stereo_sgm/depth": "/hcru0/pt_stereo_sgm.depth"
    }
    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    log_folder = "logs/DLR_20180724/"
    logfiles = [
        sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_left_image_*.msg")),
        sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_right_image_*.msg")),
        sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_sgm_depth_*.msg"))

    ]
    log_iterator = logloader.replay_files(logfiles, stream_names)

    dfc = dataflowcontrol.DataFlowControl(
        nodes={}, connections=connections, stream_aliases=stream_aliases,
        verbose=verbose)
    dfc.setup()

    app = visualization_qtgraph.ImageVisualizerApplication(
        "/hcru0/pt_stereo_sgm.depth")
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
