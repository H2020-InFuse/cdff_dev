import os
import numpy as np
from cdff_dev import (dataflowcontrol, logloader, dfnhelpers,
                      envirevisualization, path, diagrams, transformer)
from cdff_dev.dfns.imagedegradation import ImageDegradation
from cdff_dev.dfns.stereodegradation import StereoDegradation
from cdff_dev.dfns.disparityimage import DisparityImage
#from cdff_dev.dfns.disparityfiltering import DisparityFiltering
from cdff_dev.dfns.disparitytopointcloud import DisparityToPointCloud
import cdff_envire


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        transformer.EnvireDFN.__init__(self)
        self.imu_initialized = False

    def initialize_graph(self, graph):
        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.zeros(3))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
        graph.add_transform("config_sherpaTT_body", "body", t)

        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.zeros(3))
        t.transform.orientation.fromarray(np.array([0.0, 0.0, 0.0, 1.0]))
        graph.add_transform("origin", "odometry", t)

    def wheelOdometryInput(self, data):
        self._set_transform(data, frame_transformation=False)


def main():
    verbose = 0

    # Note: copy all required files in this folder, we need:
    # * converted rosbags from HCRU
    #   * recording_20181122-100414_sherpaTT_integration_0_*.msg
    #   * recording_20181122-100414_sherpaTT_integration_0_camera.msg
    #   * recording_20181122-100414_sherpaTT_integration_0_tf.msg
    # * converted pocolog from SherpaTT
    #   * sherpa_tt_mcs_Logger_InFuse.msg
    log_folder = "logs/20181122"

    prefix = "recording_20181122-100414_sherpaTT_integration_0"
    cdffpath = path.load_cdffpath()

    prefix_path = os.path.join(log_folder, prefix)

    merge_frame_pair = dfnhelpers.MergeFramePairDFN(
        left_camera_info_stream="/hcru1/pt_stereo_rect/left/camera_info",
        right_camera_info_stream="/hcru1/pt_stereo_rect/right/camera_info",
        left_is_main_camera=True, verbose=0
    )
    merge_frame_pair.set_configuration_file(prefix_path + "_camera.msg")

    transformer = Transformer()
    transformer.set_configuration_file(prefix_path + "_tf.msg")

    image_degradation = ImageDegradation()
    #image_degradation.set_configuration_file("ImageDegradationParams.yaml")
    stereo_degradation = StereoDegradation()
    #stereo_degradation.set_configuration_file("ImageDegradationParams.yaml")
    disparity_image = DisparityImage()
    #disparity_image.set_configuration_file("DisparityImageParams.yaml")
    #disparity_filtering = DisparityFiltering()
    #disparity_filtering.set_configuration_file("DisparityFilteringParams.yaml")
    disparity_to_point_cloud = DisparityToPointCloud()

    nodes = {
        "merge_frame_pair": merge_frame_pair,
        "image_degradation": image_degradation,
        "stereo_degradation": stereo_degradation,
        "disparity_image": disparity_image,
        #"disparity_filtering": disparity_filtering,
        "disparity_to_point_cloud": disparity_to_point_cloud,
        "point_cloud_filter": dfnhelpers.LambdaDFN(
            lambda x: x.filtered(), "pointCloud", "filteredPointCloud"),
        "transformer": transformer,
    }
    trigger_ports = {
        "merge_frame_pair": "rightImage",
        "image_degradation": "originalImage",
        "stereo_degradation": "originalImagePair",
        "disparity_image": "framePair",
        #"disparity_filtering": "dispImage",
        "disparity_to_point_cloud": "dispImage",
        "point_cloud_filter": "pointCloud",
        "transformer": "wheelOdometry",
    }
    connections = (
        ("/hcru1/pt_stereo_rect/left.image", "merge_frame_pair.leftImage"),
        ("/hcru1/pt_stereo_rect/right.image", "merge_frame_pair.rightImage"),
        ("/hcru1/pt_stereo_rect/left.image", "image_degradation.originalImage"),
        ("/mcs_sensor_processing.rigid_body_state_out", "transformer.wheelOdometry"),
        ("merge_frame_pair.pair", "stereo_degradation.originalImagePair"),
        ("stereo_degradation.degradedImagePair", "disparity_image.framePair"),
        ("disparity_image.rawDisparity", "disparity_to_point_cloud.dispImage"),
        #("disparity_image.dispImage", "disparity_filtering.dispImage"),
        #("disparity_filtering.filteredDispImage", "disparity_to_point_cloud.dispImage"),
        ("disparity_to_point_cloud.pointCloud", "point_cloud_filter.pointCloud"),
        #("image_degradation.degradedImage", "disparity_to_point_cloud.intensityImage"),
    )
    stream_aliases = {
        "/hcru1/pt_stereo_rect/left/image": "/hcru1/pt_stereo_rect/left.image",
        "/hcru1/pt_stereo_rect/right/image": "/hcru1/pt_stereo_rect/right.image",
        "/hcru1/pt_color/left/image": "/hcru1/pt_color/left.image",
        "/hcru1/pt_stereo_sgm/depth": "/hcru1/pt_stereo_sgm.depth",
    }

    hcru_log_iterator = logloader.replay_logfile_sequence(
        logloader.group_pattern(prefix_path, "_0*.msg"),
        ["/hcru1/pt_stereo_rect/left/image",
         "/hcru1/pt_stereo_rect/right/image",])
    sherpa_log_iterator = logloader.replay_logfile(
        os.path.join(log_folder, "sherpa_tt_mcs_Logger_InFuse.msg"),
        ["/mcs_sensor_processing.rigid_body_state_out"])
    log_iterator = logloader.replay_join(
        [hcru_log_iterator, sherpa_log_iterator])

    dfc = dataflowcontrol.DataFlowControl(
        nodes=nodes, connections=connections, trigger_ports=trigger_ports,
        stream_aliases=stream_aliases, verbose=verbose)
    dfc.setup()

    diagrams.save_graph_png(dfc, "stereo_to_pointcloud.png")

    app = envirevisualization.EnvireVisualizerApplication(
        frames={"point_cloud_filter.filteredPointCloud": "config_camera_left"},
        center_frame="origin")
    app.show_controls(log_iterator, dfc)
    app.exec_()

    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
