"""
=============================
Replay 3D Reconstruction DFPC
=============================

This script will only replay logfiles, run the DFPC and log results.
To visualize the reconstructed point cloud, you have to run

    python examples/replay_reconstruction3d_results.py
"""
print(__doc__)
import os
from cdff_dev import dataflowcontrol, logloader, path, loggermsgpack, replay
from cdff_dev.dfpcs.reconstruction3d import DenseRegistrationFromStereo


def main():
    dfc = initialize_dfc(verbose=2)
    log_iterator = initialize_log_iterator()
    replay.replay_and_process_async(
        dfc, log_iterator, queue_size=4, max_samples=None)
    dfc.node_statistics_.print_statistics()


def initialize_log_iterator():
    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    log_folder = "logs/DLR_20180724/"
    filenames = logloader.group_pattern(
        log_folder, "recording_20180724-135036_000*.msg")
    stream_names = ["/hcru0/pt_stereo_rect/left/image",
                    "/hcru0/pt_stereo_rect/right/image"]
    log_iterator = logloader.replay_logfile_sequence(filenames, stream_names)
    return log_iterator


def initialize_dfc(verbose):
    reconstruction3d = DenseRegistrationFromStereo()
    config_filename = os.path.join(
        path.load_cdffpath(),
        "Tests/ConfigurationFiles/DFPCs/Reconstruction3D/"
        "DfpcDenseRegistrationFromStereo_DlrHcru.yaml")
    reconstruction3d.set_configuration_file(config_filename)
    nodes = {"reconstruction3d": reconstruction3d}
    trigger_ports = {"reconstruction3d": "rightImage"}
    connections = (
        ("/hcru0/pt_stereo_rect/left.image", "reconstruction3d.leftImage"),
        ("/hcru0/pt_stereo_rect/right.image", "reconstruction3d.rightImage"),

        ("reconstruction3d.pointCloud", "result.pointCloud"),
        ("reconstruction3d.pose", "result.pose"),
        ("reconstruction3d.success", "result.success"),
    )
    stream_aliases = {
        "/hcru0/pt_stereo_rect/left/image": "/hcru0/pt_stereo_rect/left.image",
        "/hcru0/pt_stereo_rect/right/image": "/hcru0/pt_stereo_rect/right.image",
    }
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, trigger_ports=trigger_ports,
        stream_aliases=stream_aliases, memory_profiler=True,
        verbose=verbose)
    dfc.setup()
    logger = loggermsgpack.MsgPackLogger(
        "examples/reconstruction3d_output_log", max_samples=50,
        stream_names=["reconstruction3d.pointCloud"])
    dfc.register_logger(logger)
    return dfc


if __name__ == "__main__":
    main()
