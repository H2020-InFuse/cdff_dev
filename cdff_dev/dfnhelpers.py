import inspect

import numpy as np
import msgpack
import yaml
import pprint
import types
import cdff_types


class DFNBase:
    """Base class for data fusion nodes.

    A data fusion node does not have to inherit from it. A data fusion node
    only has to provide the same interface to be a valid DFN.
    """
    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def process(self):
        pass


class LambdaDFN:
    """A DFN that wraps a callable to implement the processing step.

    Parameters
    ----------
    lambda_fun : callable
        A function that takes one input given on an input port and produces
        one output that is returned on an output port.

    input_port : str
        Name of the input port (without 'Input' suffix)

    output_port : str
        Name of the output port (without 'Output' suffix)
    """
    def __init__(self, lambda_fun, input_port, output_port):
        self.lambda_fun = lambda_fun

        def input_port_fun(self, data):
            self.input_data = data
        input_port_fun.__name__ = input_port + "Input"

        def output_port_fun(self):
            return self.output_data
        output_port_fun.__name__ = output_port + "Output"

        setattr(self, input_port_fun.__name__,
                types.MethodType(input_port_fun, self))
        setattr(self, output_port_fun.__name__,
                types.MethodType(output_port_fun, self))

        self.input_data = None
        self.output_data = None

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def process(self):
        if self.input_data is None:
            return

        self.output_data = self.lambda_fun(self.input_data)


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


def isdfn(cls, verbose=0):
    """Check if given class is a DFN."""
    result = _check_method(cls, "set_configuration_file", verbose)
    result &= _check_method(cls, "configure", verbose)
    result &= _check_method(cls, "process", verbose)
    return result


def isdfpc(cls, verbose=0):
    """Check if a given class is a DFPC."""
    result = _check_method(cls, "set_configuration_file", verbose)
    result &= _check_method(cls, "setup", verbose)
    result &= _check_method(cls, "run", verbose)
    return result


def _check_method(cls, name, verbose=0):
    if not hasattr(cls, name):
        if verbose >= 1:
            print("Class does not have %s()" % name)
        return False
    return True


def wrap_dfpc_as_dfn(dfpc):
    """Wrap DFPC with DFN interface.

    Parameters
    ----------
    dfpc : DFPC
        DFPC object

    Returns
    -------
    dfn : DFN
        DFPC with DFN adapter
    """
    cls = create_dfn_from_dfpc(dfpc.__class__)
    return cls(dfpc=dfpc)


def create_dfn_from_dfpc(dfpc_class):
    """Create DFN adapter for DFPC.

    This is required so that a DFPC can be used in DataFlowControl.

    Parameters
    ----------
    dfpc_class : Class
        DFPC class

    Returns
    -------
    cls : Class
        DFN class
    """
    clsname = dfpc_class.__name__ + "DFN"

    def __init__(self, dfpc=dfpc_class()):
        self.dfpc = dfpc

    def set_configuration_file(self, filename):
        self.dfpc.set_configuration_file(filename)

    def configure(self):
        self.dfpc.setup()

    def process(self):
        self.dfpc.run()

    methods = {
        "__init__": __init__,
        "set_configuration_file": set_configuration_file,
        "configure": configure,
        "process": process
    }

    inputs = inspect.getmembers(dfpc_class, predicate=isinput)
    for name, _ in inputs:
        def route_method(self, data, name=name):
            getattr(self.dfpc, name)(data)
        route_method.__name__ = name
        methods[name] = route_method

    outputs = inspect.getmembers(dfpc_class, predicate=isoutput)
    for name, fun in outputs:
        def route_method(self, name=name):
            return getattr(self.dfpc, name)()
        route_method.__name__ = name
        methods[name] = route_method

    cls = type(clsname, (), methods)
    return cls


def isinput(member):
    """Test if a member function is an input port."""
    return (hasattr(member, "__name__") and
            member.__name__.endswith("Input"))


def isoutput(member):
    """Test if a member function is an output port."""
    return (hasattr(member, "__name__") and
            member.__name__.endswith("Output"))
