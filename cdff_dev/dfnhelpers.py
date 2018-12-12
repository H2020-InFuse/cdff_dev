import numpy as np
import msgpack
import yaml
import pprint
import cdff_types


class MergeFramePairDFN:
    """Merge two image streams to a stereo image.

    If they are available, camera calibrations extracted from rosbags are
    used to fill metadata of the images.

    Parameters
    ----------
    left_camera_info_stream : str, optional (default: None)
        Name of the camera info stream for the left camera

    right_camera_info_stream : str, optional (default: None)
        Name of the camera info stream for the right camera

    left_is_main_camera : bool, optional (default: None)
        If the left camera is the main camera we will assume that the
        baseline can be computed from the camera info of the right camera

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(
            self, left_camera_info_stream=None, right_camera_info_stream=None,
            left_is_main_camera=True, verbose=0):
        self.left_camera_info_stream = left_camera_info_stream
        self.right_camera_info_stream = right_camera_info_stream
        self.left_is_main_camera = left_is_main_camera
        self.verbose = verbose

        self.config_filename = None
        self.left_config = None
        self.right_config = None
        self.left_image = None
        self.right_image = None
        self.pair = cdff_types.FramePair()

    def set_configuration_file(self, filename):
        self.config_filename = filename

    def configure(self):
        if self.config_filename is None:
            if self.verbose:
                print("No configuration file given.")
            return

        with open(self.config_filename, "rb") as f:
            self.camera_configs = msgpack.unpack(
                f, encoding="utf8", use_list=False)
            self.left_config = self.camera_configs[
                self.left_camera_info_stream]
            self.right_config = self.camera_configs[
                self.right_camera_info_stream]
            if self.left_is_main_camera:
                self.pair.baseline = self.right_config["baseline"]
            else:
                self.pair.baseline = self.left_config["baseline"]

        if self.verbose:
            print("[MergeFramePairDFN] Left camera configuration:")
            print(self.left_config)
            print("[MergeFramePairDFN] Right camera configuration:")
            print(self.right_config)

    def set_time(self, time):
        pass

    def leftImageInput(self, data):
        self.left_image = data

    def rightImageInput(self, data):
        self.right_image = data

    def process(self):
        self.pair.left = self.left_image
        self.pair.right = self.right_image

        self._fill_frame_metadata(self.pair.left, self.left_config)
        self._fill_frame_metadata(self.pair.right, self.right_config)

        if self.verbose:
            str_repr = str(self.pair)
            py_repr = yaml.load(str_repr)
            pprint.pprint(py_repr, width=80, depth=4, compact=True)

    def _fill_frame_metadata(self, frame, metadata):
        if metadata is None:
            return

        frame.intrinsic.dist_coeffs.fromarray(
            np.array(metadata["intrinsic"]["distCoeffs"]))
        frame.intrinsic.camera_matrix.fromarray(
            np.array(metadata["intrinsic"]["cameraMatrix"]))
        frame.intrinsic.camera_model = metadata["intrinsic"]["cameraModel"]

        frame.extrinsic.pose_fixed_frame_robot_frame.metadata.producer_id = "MergeFramePairDFN"
        # TODO
        #frame.extrinsic.pose_fixed_frame_robot_frame.metadata.parent_frame_id = ...
        #frame.extrinsic.pose_fixed_frame_robot_frame.metadata.child_frame_id = ...
        if ("extrinsic" in metadata
                and "pose_fixed_frame_robot_frame" in metadata["extrinsic"]):
            pose_fixed_frame_robot_frame = \
                metadata["extrinsic"]["pose_fixed_frame_robot_frame"]
            frame.extrinsic.pose_fixed_frame_robot_frame.data.translation.fromarray(
                pose_fixed_frame_robot_frame["data"]["translation"])
            frame.extrinsic.pose_fixed_frame_robot_frame.data.orientation.fromarray(
                pose_fixed_frame_robot_frame["data"]["orientation"])
        else:
            frame.extrinsic.pose_fixed_frame_robot_frame.data.translation.fromarray(
                np.zeros(3))
            frame.extrinsic.pose_fixed_frame_robot_frame.data.orientation.fromarray(
                np.array([0, 0, 0, 1], dtype=np.float))

        frame.extrinsic.pose_robot_frame_sensor_frame.metadata.producer_id = "MergeFramePairDFN"
        # TODO
        #frame.extrinsic.pose_robot_frame_sensor_frame.metadata.parent_frame_id = ...
        #frame.extrinsic.pose_robot_frame_sensor_frame.metadata.child_frame_id = ...
        if ("extrinsic" in metadata
                and "pose_robot_frame_sensor_frame" in metadata["extrinsic"]):
            pose_robot_frame_sensor_frame = \
                metadata["extrinsic"]["pose_robot_frame_sensor_frame"]
            frame.extrinsic.pose_robot_frame_sensor_frame.data.translation.fromarray(
                pose_robot_frame_sensor_frame["data"]["translation"])
            frame.extrinsic.pose_robot_frame_sensor_frame.data.orientation.fromarray(
                pose_robot_frame_sensor_frame["data"]["orientation"])
        else:
            frame.extrinsic.pose_robot_frame_sensor_frame.data.translation.fromarray(
                np.zeros(3))
            frame.extrinsic.pose_robot_frame_sensor_frame.data.orientation.fromarray(
                np.array([0, 0, 0, 1], dtype=np.float))

    def pairOutput(self):
        return self.pair
