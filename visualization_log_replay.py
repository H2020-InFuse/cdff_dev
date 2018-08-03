import numpy as np
from cdff_dev import logloader, dataflowcontrol, visualization_control_panel
import cdff_types
import glob
import matplotlib.pyplot as plt
from matplotlib import animation


class LaserFilterDummyDFN:
    def __init__(self):
        self.scanSample = cdff_types.LaserScan()

    def configure(self):
        pass

    def scanSampleInput(self, scanSample):
        self.scanSample = scanSample

    def process(self):
        # filter all values that are too far away from the median
        ranges = np.array([self.scanSample.ranges[i]
                           for i in range(self.scanSample.ranges.size())])
        med = np.median(ranges)
        for i in range(len(ranges)):
            if abs(ranges[i] - med) > 100:
                self.scanSample.ranges[i] = med

    def filteredScanOutput(self):
        return self.scanSample


class PointcloudBuilderDummyDFN:
    def __init__(self):
        self.pointcloud = cdff_types.Pointcloud()

    def configure(self):
        pass

    def scanInput(self, scan):
        self.scan = scan

    def transformInput(self, transform):
        self.transform = transform

    def process(self):
        pass

    def pointcloudOutput(self):
        return self.pointcloud


class ListAssigner():
    """Handle the assignment of data to lists.
    When its add function is called, a new list and
    a corresponding entry in a dictonary are created.

    Attributes
    ----------
    _data_dict : dict
        Data types and their respective data lists

    time_length : int
        Passed from configure_time in VisualizationDataHandler. Keeps
        the current length of the time array

    data_added : bool
        Indicates whether data has been passed to the ListAssigner object from the 
        current sample
    """

    def __init__(self):
        self._data_dict = {}
        self.time_length = 0
        self.data_added = False

    def add(self, label, data, max_xrange):
        self.max_xrange = max_xrange
        if np.isnan(data):
            data = 0
        # check _data_dict for label assigned to existing list
        for key in self._data_dict:
            if key == label:
                self._data_dict[label].append(data)
                self.data_added = True

                # keep list within set size
                if len(self._data_dict[key]) > max_xrange:
                    self._data_dict[key] = self._data_dict[key][(
                        len(self._data_dict[key]) - max_xrange):len(self._data_dict[key])]
                return

        # else if label not assigned, create new list
        new_list = []

        # if list lengths don't match, fill new_list with 0's
        for i in range(self.time_length):
            new_list.append(0)

        new_list.append(data)
        if len(new_list) > max_xrange:
            new_list = new_list[(len(new_list) - max_xrange):len(new_list)]
        self._data_dict[label] = new_list
        self.data_added = True

    def remove(self, label, data_types):
        for key, _ in self._data_dict.items():
            if key == label:
                del(self._data_dict[label])
                break
        if len(self.get_labels()) != len(data_types):
            for entry in self.get_labels():
                if entry not in data_types:
                    self.remove(entry, data_types)

    def remove_all(self):
        self._data_dict.clear()

    def get_data(self):
        lists = []
        [lists.append(list_) for _, list_ in self._data_dict.items()]
        return lists

    def get_labels(self):
        labels = []
        [labels.append(key) for key, _ in self._data_dict.items()]
        return labels


class VisualizationDataHandler(dataflowcontrol.VisualizationBase):
    """Recieve log data information and control what is 
        sent to be animated.

        Attributes
        ----------
        time_list: list
            Contains all timestamps proccessed from current stream

        source_dict: dict
            Contains a reference to the source of each type of
            data. Changed based on current stream

    """

    def __init__(self):

        self.time_list = []
        self.source_dict = {}
        self.list_assigner = ListAssigner()
        self._temp = []

    def set_control_panel(self, control_panel):
        self.control_panel = control_panel

    def _configure_time(self, timestamp):
        """ Convert timestamps to seconds since first sample
        and keep timelist to correct size
        """
        self._temp.append(timestamp)
        self.time_list.append((timestamp - self._temp[0])/1000000)

        self.first_timestamp = self._temp[0]
        if len(self.time_list) >= self.max_xrange:
            self.time_list = self.time_list[(
                len(self.time_list) - self.max_xrange):len(self.time_list)]
        self.list_assigner.time_length = len(self.time_list)

    def report_node_output(self, port_name, sample, timestamp):
        """
        Attributes
        ----------
        Passed from envire_visualization_data_input:
            self.max_xrange : int
                Defines how many samples will be displayed at a time

            self.data_types : string list 
                Contains user-selected data types that will be displayed.

            self.type: string
                Contains the current user-selected sample type
        """
        self.max_xrange = self.control_panel.max_xrange
        self.data_types = self.control_panel.data_types
        self.type = self.control_panel.type
        self.stream = self.control_panel.stream

        self.list_assigner.data_added = False
        # if no data types are chosen, clear plotting lists
        if not self.data_types:
            self.list_assigner.remove_all()

        # based on what sample type is chosen, create a dictionary of data types
        elif type(sample) == cdff_types.IMUSensors and self.type == "IMUSensors" and port_name == self.stream:
            self.source_dict = {"Acceleration X": sample.acc[0], "Acceleration Y": sample.acc[1],
                                "Acceleration Z": sample.acc[2], "Gyroscopic X": sample.gyro[0],
                                "Gyroscopic Y": sample.gyro[1], "Gyroscopic Z": sample.gyro[2]}

        elif type(sample) == cdff_types.Joints and self.type == "Joints" and port_name == self.stream:
            self.source_dict = {"Position": sample.elements[0].position, "Effort": sample.elements[0].effort,
                                "Speed": sample.elements[0].speed}

        elif type(sample) == cdff_types.LaserScan and self.type == "LaserScan" and port_name == self.stream:
            self.source_dict = {"Speed": sample.speed, "Angular Resolution": sample.angular_resolution,
                                "Start Angle": sample.start_angle}

        elif type(sample) == cdff_types.RigidBodyState and self.type == "RigidBodyState" and port_name == self.stream:
            self.source_dict = {"Position X": sample.pos[0], "Position Y": sample.pos[1],
                                "Position Z": sample.pos[2]}
        else:
            self.source_dict.clear()

        #load camera frame information
        if type(sample) == cdff_types.Frame:
            #print("frame sample")
            image_array = [sample.image[i] for i in range(len(sample.image))]
            if port_name == "/camera1.frame":
                self.frame_camera1 = np.asarray(image_array, dtype=np.uint8).reshape(
                    sample.datasize.height, sample.datasize.width, 3)
            elif port_name == "/camera2.frame":
                self.frame_camera2 = np.asarray(image_array, dtype=np.uint8).reshape(
                    sample.datasize.height, sample.datasize.width, 3)

        # add data to lists to be sent to graph
        for label, data in self.source_dict.items():
            if label in self.data_types:
                self.list_assigner.add(label, data, self.max_xrange)
            else:
                self.list_assigner.remove(label, self.data_types)

        if self.list_assigner.data_added or not self.data_types:
            self._configure_time(timestamp)


def set_stream_data(log, log_img):
    """Extract stream data from given log

    Parameters
    ----------
    log: dict
        Log information loaded through logloader.py

    is_image_file: bool
        Set True if log file contains only images, False otherwise

    Returns
    -------
    stream_dict : dict
        Stream names and corresponding types
    """
    stream_dict = {}

    for stream_name in log.keys():
        if not (stream_name.endswith(".meta")):
            typename = log[stream_name + ".meta"]["type"]
            stream_dict[stream_name] = typename
    
    for stream_name in log_img.keys():
        if stream_name.endswith(".frame"):
            typename = log_img[stream_name + ".meta"]["type"]
            stream_dict[stream_name] = typename

    return stream_dict


def main(vdh, log_file):
    nodes = {
        "laser_filter": LaserFilterDummyDFN(),
        "pointcloud_builder": PointcloudBuilderDummyDFN()
    }
    periods = {
        "laser_filter": 0.025,
        "pointcloud_builder": 0.1
    }
    connections = (
        ("/hokuyo.scans", "laser_filter.scanSample"),
        ("laser_filter.filteredScan", "pointcloud_builder.scan"),
        ("/dynamixel.transforms", "pointcloud_builder.transform"),
        ("pointcloud_builder.pointcloud", "result.pointcloud")
    )

    # Logs are being loaded individually first in order to retrieve stream names
    # TODO: More efficient way of getting stream names
    log = logloader.load_log(log_file)
    logloader.print_stream_info(log)

    log_img = logloader.load_log("frames.msg")
    logloader.print_stream_info(log_img)

    stream_dict = set_stream_data(log, log_img)
    stream_names = list(stream_dict.keys())
    print("STREAMS:", stream_names)

    log_iterator = logloader.replay_files([(log_file,), ("frames.msg",)], stream_names )

    control_panel = visualization_control_panel.ControlPanelExpert(
        stream_dict)
    vdh.set_control_panel(control_panel)

    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods)
    dfc.setup()
    dfc.set_visualization(vdh)

    control_panel.show_controls(
        stream_names, log_iterator, dfc)
    control_panel.exec_()
