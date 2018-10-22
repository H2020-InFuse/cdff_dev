import os
import glob
import numpy as np
from cdff_dev import dataflowcontrol, logloader, path, envirevisualization
import cdff_envire


def main():
    verbose = 0
    stream_names = [
        "reconstruction3d.pointCloud",
    ]
    # Note that the logfiles are not in the repository because they are too
    # large. Ask Alexander Fabisch about it.
    logfiles = "examples/test_output_log_000000000.msg"
    log_iterator = logloader.replay_logfile(logfiles, stream_names)

    dfc = dataflowcontrol.DataFlowControl(
        nodes={}, connections=(), verbose=verbose)
    dfc.setup()

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
