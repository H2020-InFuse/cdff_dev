"""
========
HCRU Log
========

In this example we will replay wheel odometry data and point clouds
together with camera images and a stereo image pair.
Note that it will take a while to generate the log index.

The log data is available from Zenodo at

    TODO
"""
import os
from cdff_dev import (logloader, dataflowcontrol, envirevisualization,
                      transformer, dfnhelpers, imagevisualization)
import cdff_types


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        transformer.EnvireDFN.__init__(self)

    def initialize_graph(self, graph):
        t = transformer.make_transform(
            translation=[0.0, 0.0, 0.442],
            orientation=[0.0, 0.0, 0.0, 1.0])
        graph.add_transform("body", "velodyne_plane_fixed", t)

        t = transformer.make_transform(
            translation=[-0.12889, -0.01697, 0.09081],
            orientation=[0.0, 0.0, 0.06540328, 0.99785891])
        graph.add_transform("velodyne_plane_moving", "velodyne", t)

        t = transformer.make_transform(
            translation=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0])
        graph.add_transform("config_sherpaTT_body", "body", t)

        t = transformer.make_transform(
            translation=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0])
        graph.add_transform("origin", "odometry", t)

    def wheelOdometryInput(self, data):
        self._set_transform(data, frame_transformation=False)

    def bodyJointInput(self, data):
        self._set_transform(data, frame_transformation=False)

    def odometryTrajectoryOutput(self):
        if self.graph_.contains_edge("odometry", "body"):
            position = cdff_types.Vector3d()
            body2odometry = self._get_transform(
                "odometry", "body", frame_transformation=True)
            position.fromarray(body2odometry.pos.toarray())
            return position
        else:
            return None


def replay_logfile_join(log_folder, logfiles):  # shortcut
    return logloader.replay_join([
        logloader.replay_logfile(
            os.path.join(log_folder, filename), stream_name)
        for filename, stream_name in logfiles
    ])


def main():
    log_folder = "logs/open_sherpatt_bremen"
    sherpa_log_iterator = replay_logfile_join(
        log_folder,
        [("sherpa_tt_mcs_Logger_InFuse.msg", ["/mcs_sensor_processing.rigid_body_state_out"]),
         ("body_joint_Logger_InFuse.msg", ["/body_joint.body_joint_samples"]),
         ("sherpa_tt_slam_Logger_InFuse.msg", ["/slam_filter.output"])]
    )

    log_folder = "logs/open_hcru_bremen"
    prefix = "recording_20180927-175540_sherpaTT_integration"
    prefix_path = os.path.join(log_folder, prefix)
    hcru_log_iterator = logloader.replay_logfile_sequence(
        logloader.group_pattern(prefix_path, "_0*.msg"),
        ["/hcru1/pt_stereo_rect/left/image",
         "/hcru1/pt_stereo_rect/right/image",
         #"/hcru1/pt_color/left/image",
         #"/hcru1/pt_stereo_sgm/depth"
         ])

    log_iterator = logloader.replay_join(
        [sherpa_log_iterator, hcru_log_iterator])

    stream_aliases = {
        "/hcru1/pt_stereo_rect/left/image": "/hcru1/pt_stereo_rect/left.image",
        "/hcru1/pt_stereo_rect/right/image": "/hcru1/pt_stereo_rect/right.image",
        #"/hcru1/pt_color/left/image": "/hcru1/pt_color/left.image",
        #"/hcru1/pt_stereo_sgm/depth": "/hcru1/pt_stereo_sgm.depth",
    }

    app = envirevisualization.EnvireVisualizerApplication(
        frames={
            "/slam_filter.output": "velodyne",
            "transformer.odometryTrajectory": "origin",
        },
        urdf_files=[],
        center_frame="odometry"
    )

    transformer = Transformer()
    transformer.set_configuration_file(prefix_path + "_tf.msg")

    merge_frame_pair = dfnhelpers.MergeFramePairDFN(
        left_camera_info_stream="/hcru1/pt_stereo_rect/left/camera_info",
        right_camera_info_stream="/hcru1/pt_stereo_rect/right/camera_info",
        left_is_main_camera=True, verbose=0
    )
    merge_frame_pair.set_configuration_file(prefix_path + "_camera.msg")

    dfc = dataflowcontrol.DataFlowControl(
        nodes={"transformer": transformer,
               "merge_frame_pair": merge_frame_pair},
        connections=(
            ("/mcs_sensor_processing.rigid_body_state_out", "transformer.wheelOdometry"),
            ("/body_joint.body_joint_samples", "transformer.bodyJoint"),
            ("/hcru1/pt_stereo_rect/left.image", "merge_frame_pair.leftImage"),
            ("/hcru1/pt_stereo_rect/right.image", "merge_frame_pair.rightImage"),
        ),
        periods={"transformer": 1,
                 "merge_frame_pair": 1},
        real_time=False,
        stream_aliases=stream_aliases,
        verbose=0
    )
    dfc.setup()

    from cdff_dev.diagrams import save_graph_png
    save_graph_png(dfc, "hcru_sherpa.png")

    visualization = imagevisualization.ImagePairVisualization(
        "merge_frame_pair.pair")
    dfc.register_visualization(visualization)

    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
