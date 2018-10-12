import glob
import os
from cdff_dev import dataflowcontrol, logloader, imagevisualization


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
    log_folder = os.path.expanduser("~") + "/Research/projects/ongoing/EU-OG3_InFUSE_18488/documentation/experiments/20180927_sherpa_and_hcru_log/converted2"
    prefix = "recording_20180927-175146_sherpaTT_integration"

    prefix_path = os.path.join(log_folder, prefix)

    def group_pattern(prefix_path, pattern):
        files = glob.glob(prefix_path + pattern)
        if len(files) == 0:
            dirname = os.sep.join(prefix_path.split(os.sep)[:-1])
            if not os.path.exists(dirname):
                raise ValueError("Directory '%s' does not exist" % dirname)
            actual_files = glob.glob(dirname + "*")
            raise ValueError("Could not find any matching files, only found %s"
                             % actual_files)
        return sorted(files)

    log_iterator = logloader.replay_join([
        logloader.replay_logfile(
            filename,
            ["/hcru1/pt_stereo_rect/left/image",
             "/hcru1/pt_stereo_rect/right/image",
             "/hcru1/pt_color/left/image",
             "/hcru1/pt_stereo_sgm/depth"]
        )
        for filename in group_pattern(prefix_path, "_0*.msg")
    ])

    dfc = dataflowcontrol.DataFlowControl(
        nodes={}, connections=connections, stream_aliases=stream_aliases,
        verbose=verbose)
    dfc.setup()

    app = imagevisualization.ImageVisualizerApplication("/hcru1/pt_stereo_rect/left.image")
    #app = imagevisualization.ImageVisualizerApplication("/hcru1/pt_stereo_rect/right.image")
    #app = imagevisualization.ImageVisualizerApplication("/hcru1/pt_color/left.image")
    #app = imagevisualization.ImageVisualizerApplication("/hcru1/pt_stereo_sgm.depth")
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
