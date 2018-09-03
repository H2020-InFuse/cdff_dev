import glob
from cdff_dev import dataflowcontrol, visualization2d


def main():
    verbose = 2
    nodes = {
        # TODO
    }
    periods = {
        # TODO
    }
    connections = (
        ("/hcru0.pose_cov", "result.pose"),
        ("/hcru0/pt_stereo_rect/left.image", "result.left_image"),
        ("/hcru0/pt_stereo_rect/right.image", "result.right_image"),
    )

    stream_names = [
        "/hcru0/pose_cov",
        "/hcru0/pt_stereo_rect/left/image",
        "/hcru0/pt_stereo_rect/right/image",
    ]
    stream_aliases = {
        "/hcru0/pose_cov": "/hcru0.pose_cov",
        "/hcru0/pt_stereo_rect/left/image": "/hcru0/pt_stereo_rect/left.image",
        "/hcru0/pt_stereo_rect/right/image": "/hcru0/pt_stereo_rect/right.image",
    }
    image_stream_names = [
        "/hcru0/pt_stereo_rect/left.image",
        "/hcru0/pt_stereo_rect/right.image"
    ]
    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    log_folder = "logs/converted/"
    logfiles = [
        [log_folder + "recording_20180724-135036_hcru0_pose_cov_%09d.msg" % i for i in range(5)],
        #sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pose_cov_*.msg")),
        [log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_left_image_%09d.msg" % i for i in range(2)],
        #sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_left_image_*.msg")),
        [log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_right_image_%09d.msg" % i for i in range(2)],
        #sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_right_image_*.msg"))
    ]

    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods, stream_aliases=stream_aliases,
        verbose=verbose)
    dfc.setup()

    vis = visualization2d.MatplotlibVisualizerApplication()
    vis.show_controls(
        dfc, logfiles, stream_names, image_stream_names=image_stream_names,
        image_shape=(1032, 772))
    vis.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
