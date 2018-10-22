import os
import glob
import numpy as np
from cdff_dev import (dataflowcontrol, logloader, path, envirevisualization,
                      loggermsgpack)
from cdff_dev.dfpcs.reconstruction3d import EstimationFromStereo
import cdff_envire


def main():
    verbose = 0
    reconstruction3d = EstimationFromStereo()
    config_filename = os.path.join(
        path.load_cdffpath(),
        "Tests/ConfigurationFiles/DFPCs/Reconstruction3D/"
        "DfpcEstimationFromStereo_DlrHcru.yaml")
    reconstruction3d.set_configuration_file(config_filename)
    nodes = {
        "reconstruction3d": reconstruction3d
    }
    trigger_ports = {
        "reconstruction3d": "rightImage"
    }
    connections = (
        ("/hcru0/pt_stereo_rect/left.image", "reconstruction3d.leftImage"),
        ("/hcru0/pt_stereo_rect/right.image", "reconstruction3d.rightImage"),

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
        sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_left_image_*.msg")),
        sorted(glob.glob(log_folder + "recording_20180724-135036_hcru0_pt_stereo_rect_right_image_*.msg"))
    ]
    log_iterator = logloader.replay_files(logfiles, stream_names)

    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, trigger_ports=trigger_ports,
        stream_aliases=stream_aliases, verbose=verbose)
    dfc.setup()
    dfc.register_logger(loggermsgpack.MsgPackLogger("examples/test_output_log"))

    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            "reconstruction3d.pointCloud": "camera",
        },
        urdf_files=[],
        center_frame="center"
    )
    graph = app.visualization.world_state_.graph_
    t = cdff_envire.Transform()
    t.transform.translation.fromarray(np.array([0, 0, 1.5]))
    #pr.quaternion_xyzw_from_wxyz(pr.quaternion_from_matrix(pr.matrix_from_euler_xyz([pi, 0, pi / 2]).dot(pr.matrix_from_euler_xyz([0, pi / 2, pi / 4]))))
    t.transform.orientation.fromarray(np.array([0.27059805,  0.65328148, -0.65328148,  0.27059805]))
    graph.add_transform("camera", "center", t)

    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
