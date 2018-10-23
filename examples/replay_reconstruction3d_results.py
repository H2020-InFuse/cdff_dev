"""
===================================
Visualize Reconstructed Point Cloud
===================================

This script will load results of the 3D reconstruction DFPC and display them
in the EnviRe visualizer.
Note that the logfiles are not in the repository because they are too
large. You can generate them with

    python examples/replay_reconstruction3d.py
"""
print(__doc__)
import numpy as np
from cdff_dev import dataflowcontrol, logloader, envirevisualization
import cdff_envire


def main():
    log_iterator = logloader.replay_logfile_sequence(
        logloader.group_pattern(
            "examples/reconstruction3d_output_log", "_*.msg"),
        ["reconstruction3d.pointCloud"])

    dfc = dataflowcontrol.DataFlowControl(nodes={}, connections=())
    dfc.setup()

    app = envirevisualization.EnvireVisualizerApplication(
        frames={"reconstruction3d.pointCloud": "camera"}, center_frame="center")

    graph = app.visualization.world_state_.graph_
    t = cdff_envire.Transform()
    t.transform.translation.fromarray(np.array([0, 0, 1.5]))
    #pr.quaternion_xyzw_from_wxyz(pr.quaternion_from_matrix(pr.matrix_from_euler_xyz([pi, 0, pi / 2]).dot(pr.matrix_from_euler_xyz([0, pi / 2, pi / 4]))))
    t.transform.orientation.fromarray(
        np.array([0.27059805,  0.65328148, -0.65328148,  0.27059805]))
    graph.add_transform("camera", "center", t)

    app.show_controls(log_iterator, dfc)
    app.exec_()


if __name__ == "__main__":
    main()
