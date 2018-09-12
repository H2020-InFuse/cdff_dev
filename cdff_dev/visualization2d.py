import sys
import threading
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import SpanSelector
from matplotlib.colors import NoNorm
from . import logloader, dataflowcontrol
from .envirevisualization import Worker, Step  # TODO move to another file
import cdff_types


class MatplotlibVisualizerApplication:
    def show_controls(self, dfc, logfiles, stream_names,
                      image_stream_names=(), image_shape=(0, 0),
                      stream_aliases=None):
        # TODO start everything in main thread
        if stream_aliases is None:
            stream_aliases = {}

        self.vdh = VisualizationDataHandler(image_stream_names)

        # create figure, subplots
        self.fig = plt.figure()

        self.ax = None
        self.ax_images = []

        self.images = []

        if not image_stream_names:
            self.ax = plt.subplot(111)
        else:
            n_images = len(image_stream_names)
            if n_images > 2:
                raise NotImplementedError(
                    "Only one or two images are supported, trying to plot %d."
                    % n_images)

            self.ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
            blank_image = np.empty(image_shape, dtype=np.uint8)
            if len(image_shape) == 2:
                cmap = "gray"
                blank_image[:, :] = 255
            else:
                cmap = None
                blank_image[:, :, :] = 255
            for i in range(n_images):
                ax_image = plt.subplot2grid((2, 2), (1, i))
                ax_image.xaxis.set_major_locator(plt.NullLocator())
                ax_image.yaxis.set_major_locator(plt.NullLocator())
                self.ax_images.append(ax_image)
                image = ax_image.imshow(
                    blank_image, cmap=cmap, norm=NoNorm(), animated=True)
                self.images.append(image)

        # set axis titles
        self.ax.set_ylabel("Units")
        self.ax.set_xlabel("Time (seconds)")

        # format axis labels
        xfmt = ScalarFormatter(useMathText=True)
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(10))
        self.ax.yaxis.set_major_formatter(xfmt)
        self.ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(10))

        plt.tight_layout()

        self.span = SpanSelector(
            self.ax, self.vdh.onselect, 'horizontal', useblit=True,
            rectprops=dict(alpha=0.5, facecolor='red'))

        # TODO removes blinking from selection, but data is not displayed
        #def onclick(event):
        #    self.anim.event_source.stop()
        #cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

        # This is the only part that limits how many types of data
        # can be graphed at once
        line0, = self.ax.plot([], color="b", linewidth=0.5, alpha=0.75)
        line1, = self.ax.plot([], color="r", linewidth=0.5, alpha=0.75)
        line2, = self.ax.plot([], color="g", linewidth=0.5, alpha=0.75)
        line3, = self.ax.plot([], color="darkorange", linewidth=0.5, alpha=0.75)
        marker, = self.ax.plot([], color="k", marker="+")
        self.line = [line0, line1, line2, line3, marker]

        # Setting blit to False renders animation useless - only grey window
        # displayed. Blitting may have some connection to removing blinking
        # from animation.
        fargs = self.line, self.ax, self.vdh, self.images, self.ax_images
        self.anim = animation.FuncAnimation(
            self.fig, animate, fargs=fargs, interval=0.0, blit=True)

        # begin log replay on a separate thread in order to run concurrently
        thread = threading.Thread(
            target=main,
            args=(dfc, self.vdh, logfiles, stream_names, image_stream_names,
                  stream_aliases))
        thread.start()

    def exec_(self):
        # display plotting window
        try:
            plt.show()
            #matplotlib.interactive(True)
            #plt.draw()
            #plt.pause(0.001)
        except AttributeError:
            pass
        plt.close()


def animate(i, line, ax, vdh, images=None, ax_images=None, plot_frequency=75):
    """Update and plot line data.

    Anything that needs to be updated must lie within this function.
    It is repeatedly called by FuncAnimation, incrementing i with each
    iteration.
    """
    redraw = i % plot_frequency == 0

    times = vdh.time_list
    list_assigner = vdh.list_assigner
    data_lists = list_assigner.get_data()
    data_labels = list_assigner.get_labels()

    # pass data from lists to lines
    if data_lists:
        vdh.set_axis_limits(ax, data_lists)

        try:
            for index, measurements in enumerate(data_lists):
                if len(measurements) != len(times):
                    raise ValueError(
                        "Y data and X data have different lengths. X: %d; Y: %d"
                        % (len(times), len(measurements)))

                line[index].set_data(times[0:i], measurements[0:i])
        except ValueError as e:
            print(e)

        try:
            if vdh.control_panel.remove_outlier:
                delete_last_line(vdh)
                print("line deleted")
        except AttributeError:
            pass

        plt.legend(handles=line, labels=data_labels, fancybox=False,
                   frameon=True, loc="best")

    if redraw and images and ax_images:
        for image, image_data in zip(images, vdh.images):
            if image_data is not None:
                image.set_data(image_data)

    # The best solution for displaying "animated" tick labels.
    # A better solution would be to selectively only redraw these labels,
    # instead of the entire plot
    if redraw:
        plt.draw()

    return line


def delete_last_line(vdh):
    file_r = open(vdh.control_panel.outlier_file(), "r")
    lines = file_r.readlines()

    file_w = open(vdh.control_panel.outlier_file(), "w")
    file_w.writelines(lines[:-1])
    vdh.control_panel.remove_outlier = False


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


#TODO: better name
class Coordinates():
    """Instantiate and maintain basic coordinate aspects of the plot.

    These points are stored for use by other functions.

    Attributes
    ---------
    span_selected : list
        stores the range of X-values selected by the user from onselect function

    data_min/data_max : float
        min/max value from Y-Axis data
    """
    def __init__(self):
        self.span_selected = None
        self.data_min = float("inf")
        self.data_max = -float("inf")

    def yrange_reset(self):
        self.data_min = float("inf")
        self.data_max = -float("inf")


class VisualizationDataHandler(dataflowcontrol.VisualizationBase):
    """Recieve log data information and control what is sent to be animated.

    Parameters
    ----------
    image_stream_names : list, optional (default: [])
        Name of the image data streams

    Attributes
    ----------
    time_list: list
        Contains all timestamps proccessed from current stream

    source_dict: dict
        Contains a reference to the source of each type of
        data. Changed based on current stream

    Passed from envire_visualization_data_input:
    self.max_xrange : int
        Defines how many samples will be displayed at a time

    self.data_types : string list
        Contains user-selected data types that will be displayed.

    self.type: string
        Contains the current user-selected sample type
    """
    def __init__(self, image_stream_names=()):
        self.time_list = []
        self.source_dict = {}
        self.list_assigner = ListAssigner()
        self._temp = []
        self.image_stream_names = image_stream_names
        self.images = [None] * len(self.image_stream_names)
        self.coords = Coordinates()

    def set_control_panel(self, control_panel):
        self.control_panel = control_panel

    def _set_time(self, timestamp):
        """Convert timestamps to seconds since first sample.

        Also keep time list to correct size.
        """
        self._temp.append(timestamp)
        self.time_list.append((timestamp - self._temp[0]) / 1e6)

        self.first_timestamp = self._temp[0]
        if len(self.time_list) >= self.max_xrange:
            self.time_list = self.time_list[(
                len(self.time_list) - self.max_xrange):len(self.time_list)]
        self.list_assigner.time_length = len(self.time_list)

    def report_node_output(self, port_name, sample, timestamp):
        self.max_xrange = self.control_panel.max_xrange
        self.data_types = self.control_panel.data_types
        self.type = self.control_panel.type
        self.stream = self.control_panel.stream

        self.list_assigner.data_added = False
        # if no data types are chosen, clear plotting lists
        if not self.data_types:
            self.list_assigner.remove_all()

        # based on what sample type is chosen, create a dictionary of data types
        elif (type(sample) == cdff_types.IMUSensors and
              self.type == "IMUSensors" and
              port_name == self.stream):
            self.source_dict = {
                "Acceleration X": sample.acc[0],
                "Acceleration Y": sample.acc[1],
                "Acceleration Z": sample.acc[2],
                "Gyroscopic X": sample.gyro[0],
                "Gyroscopic Y": sample.gyro[1],
                "Gyroscopic Z": sample.gyro[2]
            }

        elif (type(sample) == cdff_types.Joints and
              self.type == "Joints" and
              port_name == self.stream):
            self.source_dict = {
                "Position": sample.elements[0].position,
                "Effort": sample.elements[0].effort,
                "Speed": sample.elements[0].speed
            }

        elif (type(sample) == cdff_types.LaserScan and
              self.type == "LaserScan" and
              port_name == self.stream):
            self.source_dict = {
                "Speed": sample.speed,
                "Angular Resolution": sample.angular_resolution,
                "Start Angle": sample.start_angle
            }

        elif (type(sample) == cdff_types.RigidBodyState and
              self.type == "RigidBodyState" and
              port_name == self.stream):
            self.source_dict = {
                "Position X": sample.pos[0],
                "Position Y": sample.pos[1],
                "Position Z": sample.pos[2]
            }
        else:
            self.source_dict.clear()

        # load camera frame information
        if (port_name in self.image_stream_names and
                type(sample) in [cdff_types.Image, cdff_types.Frame]):
            image_idx = self.image_stream_names.index(port_name)
            self.images[image_idx] = sample.array_reference().copy()

        # add data to lists to be sent to graph
        for label, data in self.source_dict.items():
            if label in self.data_types:
                self.list_assigner.add(label, data, self.max_xrange)
            else:
                self.list_assigner.remove(label, self.data_types)

        if self.list_assigner.data_added or not self.data_types:
            self._set_time(timestamp)


    def set_axis_limits(self, ax, data_lists):
        """Control axis limits that change dynamically with data."""
        control_panel = self.control_panel

        # find highest and lowest data points out of all lists given
        for the_list in data_lists:
            self.coords.data_min = min(np.amin(the_list), self.coords.data_min)
            self.coords.data_max = max(np.amax(the_list), self.coords.data_max)

        # set x and y limits based on current highest and lowest data points
        if data_lists:
            margin = max(np.abs(self.coords.data_min),
                         self.coords.data_max) * 0.05
            ax.set_ylim(self.coords.data_min - margin,
                        self.coords.data_max + margin)

        if self.source_dict and self.control_panel.stream_reset:
            self.coords.yrange_reset()
            control_panel.stream_reset = False

        if self.control_panel.yrange_reset:
            self.coords.yrange_reset()
            control_panel.yrange_reset = False

        ax.set_xlim(self.time_list[0], np.amax(self.time_list))

    def convert_back_timestamp(self, timestamp):
        return int(timestamp * 1e6) + self.first_timestamp

    def onselect(self, xmin, xmax):
        """When a range is selected, prints selected range to file."""
        file_ = open(self.control_panel.outlier_file, "a+")

        indmin, indmax = np.searchsorted(self.time_list, (xmin, xmax))
        indmax = min(len(self.time_list) - 1, indmax)

        self.coords.span_selected = self.time_list[indmin:indmax]

        #TODO: simplify file write - string join
        file_.write("[")
        try:
            for i, num in enumerate(self.coords.span_selected):
                if i != len(self.coords.span_selected) - 1:
                    file_.write("%d, " % (self.convert_back_timestamp(num)))
                else:
                    file_.write("%d]\n" % (self.convert_back_timestamp(num)))
        except TypeError:
            print("Data is not in list form")
            pass

        file_.close()


def main(dfc, vdh, log_files, stream_names, image_stream_name, stream_aliases,
         verbose=0):
    typenames = logloader.summarize_logfiles(log_files)
    if verbose:
        print("Streams: %s" % stream_names)
        if image_stream_name is not None:
            print("Image stream: % s" % image_stream_name)

    log_iterator = logloader.replay_files(log_files, stream_names)

    control_panel = ControlPanelExpert(typenames)
    vdh.set_control_panel(control_panel)

    dfc.set_visualization(vdh)

    control_panel.show_controls(stream_names, log_iterator, dfc, stream_aliases)
    control_panel.exec_()


class ControlPanelExpert:
    """Qt Application.

    Parameters
    ----------
    stream_dict : dict
        All stream names and types

    Attributes
    ----------
    data_types : list
        All currently selected data types

    stream : str
        Currently selected stream

    outlier_file : str
        File that outlier selection is sent to

    stream_reset : bool
        Indicates whether a new stream has been chosen

    max_xrange : int
        Number of samples displayed on graph at a time

    yrange_reset : bool
        Indicates whether the vertical axis has been reset

    type : str
        Currently selected stream's type (IMUSensors, Laserscan, etc)

    remove_outlier : bool
        Indicates whether the last selected outlier span should be removed from file
    """

    def __init__(self, stream_dict):
        self.app = QApplication(sys.argv)
        self.data_types = []
        self.stream = None
        self.outlier_file = "outliers.txt"
        self.stream_dict = stream_dict
        self.stream_reset = False
        self.max_xrange = 100
        self.yrange_reset = False
        self.type = None
        self.remove_outlier = False

    def show_controls(self, stream_names, log_iterator, dfc, stream_aliases):
        """Show control window to replay log file.

        Parameters
        ----------
        stream_names : list
            List of stream names available to be replayed

        log_iterator : generator
            Log data

        dfc : DataFlowControl
            Configured processing and data fusion logic

        stream_aliases : dict, optional (default: {})
            Mapping from original stream names to their aliases if they have
            any.
        """
        # splits args between image log data and other
        self.control_window = ControlPanelMainWindow(
            Step, self, self.stream_dict, stream_names, log_iterator, dfc,
            stream_aliases)
        self.control_window.show()

    def exec_(self):
        """Start Qt application.

        Qt will take over the main thread until the main window is closed.
        """
        self.app.exec_()


class ControlPanelMainWindow(QMainWindow):
    """Instantiation of Qt window and Control Panel classes."""
    def __init__(self, work, ctrl_pnl_expert, stream_dict, stream_names,
                 log_iterator, dfc, stream_aliases):
        super(ControlPanelMainWindow, self).__init__()
        self.worker = Worker(work, log_iterator, dfc)
        self.worker.start()
        QApplication.instance().aboutToQuit.connect(self.worker.quit)

        self.setWindowTitle("Visualization Control Panel")
        self.setMinimumSize(400, 310)

        central_widget = ControlPanelWidget(
            self.worker, ctrl_pnl_expert, stream_dict, stream_names,
            stream_aliases)

        controller = ControlPanelController(
            central_widget, self.worker, ctrl_pnl_expert)

        central_widget.set_controller(controller)
        central_widget.add_widgets()
        self.setCentralWidget(central_widget)

    def closeEvent(self, QCloseEvent):
        self.worker.quit()


class ControlPanelWidget(QWidget):
    """Contains widgets that comprise physical layout of control panel.

    Parameters
    ---------
    worker : Worker
        Controls iteration of Step class

    ctrl_pnl_expert : ControlPanelExpert
        Contains all necessary data values that are accessed externally

    stream_dict : dict
        All stream names and types

    stream_names : list
        List of stream names available to be replayed

    stream_aliases : dict, optional (default: {})
        Mapping from original stream names to their aliases if they have any.
    """
    def __init__(self, worker, ctrl_pnl_expert, stream_dict, stream_names,
                 stream_aliases):
        super(ControlPanelWidget, self).__init__()
        self.stream_names = stream_names
        self.stream_dict = stream_dict
        self.stream_aliases = stream_aliases
        self.ctrl_pnl_expert = ctrl_pnl_expert
        self.worker = worker

    def add_widgets(self):
        layout_base = QHBoxLayout()
        layout_sub_1 = QVBoxLayout()
        layout_sub_2 = QVBoxLayout()

        layout_base.addLayout(layout_sub_1)
        layout_base.addLayout(layout_sub_2)
        self.setLayout(layout_base)

        # Replay controls label
        replay_tools = QLabel("Replay Controls")
        replay_tools.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        layout_sub_1.addWidget(replay_tools)

        # Step
        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.worker.step)
        self.step_button.setMinimumSize(200, 0)
        layout_sub_1.addWidget(self.step_button)

        # Play
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.worker.play)
        layout_sub_1.addWidget(self.play_button)

        # Pause
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.worker.pause)
        layout_sub_1.addWidget(self.pause_button)

        # Break label
        break_label = QLabel("Break between samples:")
        break_label.setAlignment(Qt.AlignBottom)
        layout_sub_1.addWidget(break_label)

        # Break between samples
        self.break_edit = QDoubleSpinBox()
        self.break_edit.setValue(self.worker.break_length_)
        self.break_edit.setMinimum(0.00)
        self.break_edit.setMaximum(sys.float_info.max)
        self.break_edit.setSingleStep(0.001)
        self.break_edit.setDecimals(3)
        self.break_edit.valueChanged.connect(self.controller.update_break)
        layout_sub_1.addWidget(self.break_edit)

        # Current sample label
        c_sample = QLabel("Current sample:")
        c_sample.setAlignment(Qt.AlignBottom)
        layout_sub_1.addWidget(c_sample)

        # Current sample
        self.step_info = QSpinBox()
        self.step_info.setMaximum(2 ** 31 - 1)
        self.step_info.setEnabled(False)
        layout_sub_1.addWidget(self.step_info)

        layout_sub_1.addWidget(QLabel(""))

        # Data Selection label
        data_tools = QLabel("Data Selection")
        data_tools.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        layout_sub_2.addWidget(data_tools)

        # Stream select
        self.stream_edit = QComboBox()
        self.stream_edit.addItem("Select stream...")
        stream_names = [self.stream_aliases.get(sn, sn)
                        for sn in self.stream_names]
        self.stream_edit.addItems(stream_names)
        self.stream_edit.currentIndexChanged.connect(
            self.controller.set_stream)
        layout_sub_2.addWidget(self.stream_edit)

        # Data select
        self.data_edit = QComboBox()
        self.data_edit.addItems(["No Data"])
        layout_sub_2.addWidget(self.data_edit)

        h_box = QHBoxLayout()
        layout_sub_2.addLayout(h_box)

        # Add data
        self.data_add_button = QPushButton("Add Data")
        self.data_add_button.clicked.connect(self.controller.add_data)
        h_box.addWidget(self.data_add_button)

        # Remove data
        self.data_rem_button = QPushButton("Remove Data")
        self.data_rem_button.clicked.connect(self.controller.remove_data)
        h_box.addWidget(self.data_rem_button)

        # Number of samples displayed label
        xrange_label = QLabel("# of samples displayed:")
        xrange_label.setAlignment(Qt.AlignBottom)
        layout_sub_2.addWidget(xrange_label)

        h_box1 = QHBoxLayout()
        layout_sub_2.addLayout(h_box1)

        # Number of samples displayed
        self.xrange_edit = QDoubleSpinBox()
        self.xrange_edit.setValue(100)
        self.xrange_edit.setMinimum(10)
        self.xrange_edit.setMaximum(sys.float_info.max)
        self.xrange_edit.setSingleStep(10)
        self.xrange_edit.setDecimals(0)
        self.xrange_edit.valueChanged.connect(
            self.controller.update_max_xrange)
        h_box1.addWidget(self.xrange_edit)

        # Reset Y limits
        self.yrange_reset_button = QPushButton("Reset Y Limits")
        self.yrange_reset_button.clicked.connect(self.controller.reset_yrange)
        h_box1.addWidget(self.yrange_reset_button)

        # Annotation tools label
        tools = QLabel("Annotation Tools")
        tools.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        layout_sub_2.addWidget(tools)

        # Outlier file select
        self.outlier_file_name = QLineEdit()
        self.outlier_file_name.setPlaceholderText("Save outliers to...")
        self.outlier_file_name.returnPressed.connect(
            self.controller.set_outlier_file)
        layout_sub_2.addWidget(self.outlier_file_name)

        # Remove last selection
        self.outlier_remove_button = QPushButton("Undo last select")
        self.outlier_remove_button.clicked.connect(
            self.controller.remove_outlier)
        layout_sub_2.addWidget(self.outlier_remove_button)

        self.worker.step_done.connect(self.step_info.setValue)

    def set_controller(self, controller):
        self.controller = controller


class ControlPanelController():
    """ Contains all functions for manipulating the visualization.

    Configuration is then passed to the ControlPanelExpert.

    Parameters
    ---------
    central_widget : ControlPanelWidget
        Holds all widget instantiations and positioning

    worker : Worker
        Controls iteration of Step class

    ctrl_pnl_expert : ControlPanelExpert
        Contains all necessary data values that are accessed externally
    """
    def __init__(self, central_widget, worker, ctrl_pnl_expert):
        self.central_widget = central_widget
        self.worker = worker
        self.ctrl_pnl_expert = ctrl_pnl_expert

    def update_break(self):
        try:
            break_length = round(self.central_widget.break_edit.value(),
                                 self.central_widget.break_edit.decimals())
            if break_length < 0:
                raise ValueError("Length smaller than 0: %g" % break_length)
        except ValueError:
            print("Invalid break length: '%f'" % break_length)
        self.worker.break_length_ = break_length

    def _trigger_reset(self):
        self.ctrl_pnl_expert.stream_reset = True
        self.ctrl_pnl_expert.data_types = []
        self.central_widget.data_edit.clear()

    def set_stream(self):
        stream_choice = self.central_widget.stream_edit.currentText()
        if stream_choice != self.ctrl_pnl_expert.stream and stream_choice != "Select stream...":
            self.ctrl_pnl_expert.stream = stream_choice
            for stream, type_ in self.central_widget.stream_dict.items():
                if stream == stream_choice:
                    self.set_data_options(type_)
                    break

    def set_data_options(self, type_):
        if type_ == "IMUSensors":
            self.ctrl_pnl_expert.type = "IMUSensors"
            self._trigger_reset()
            self.central_widget.data_edit.addItems(
                ["Select data...",
                 "Acceleration X", "Acceleration Y", "Acceleration Z",
                 "Gyroscopic X", "Gyroscopic Y", "Gyroscopic Z"])

        elif type_ == "Joints":
            self.ctrl_pnl_expert.type = "Joints"
            self._trigger_reset()
            self.central_widget.data_edit.addItems(
                ["Select data...", "Position", "Effort", "Speed"])

        elif type_ == "LaserScan":
            self.ctrl_pnl_expert.type = "LaserScan"
            self._trigger_reset()
            self.central_widget.data_edit.addItems(
                ["Select data...", "Speed", "Angular Resolution", "Start Angle"])

        elif type_ == "RigidBodyState":
            self.ctrl_pnl_expert.type = "RigidBodyState"
            self._trigger_reset()
            self.central_widget.data_edit.addItems(
                ["Select data...", "Position X", "Position Y", "Position Z"])

        else:
            self._trigger_reset()
            self.central_widget.data_edit.addItem("No Data")

    def add_data(self):
        data_choice = self.central_widget.data_edit.currentText()
        if len(self.ctrl_pnl_expert.data_types) == 4:
            print("Graph Display Max Reached")
        elif ((data_choice not in self.ctrl_pnl_expert.data_types) and
              ("+" not in data_choice) and (data_choice != "No Data") and
              (data_choice != "Select data...")):
            self.ctrl_pnl_expert.data_types.append(data_choice)
            self.central_widget.data_edit.setItemText(
                self.central_widget.data_edit.currentIndex(), (data_choice + " +"))

    def remove_data(self):
        data_choice = self.central_widget.data_edit.currentText()[:-2]
        if (data_choice in self.ctrl_pnl_expert.data_types) and (data_choice != "Select data..."):
            self.ctrl_pnl_expert.data_types.remove(data_choice)
            self.central_widget.data_edit.setItemText(
                self.central_widget.data_edit.currentIndex(), data_choice)

    def update_max_xrange(self):
        value = self.central_widget.xrange_edit.value()
        try:
            max_xrange = float(value)
            if max_xrange < 0:
                raise ValueError("Width smaller than 0: %g" % max_xrange)
        except ValueError:
            print("Invalid viewport width: '%f'" % value)
        self.ctrl_pnl_expert.max_xrange = (int(value))

    def reset_yrange(self):
        self.ctrl_pnl_expert.yrange_reset = True

    def set_outlier_file(self):
        if self.central_widget.outlier_file_name.text() == "":
            self.ctrl_pnl_expert.outlier_file = "outliers.txt"
        else:
            self.ctrl_pnl_expert.outlier_file = self.central_widget.outlier_file_name.text()
        print(self.ctrl_pnl_expert.outlier_file)

    def remove_outlier(self):
        self.ctrl_pnl_expert.remove_outlier = True
