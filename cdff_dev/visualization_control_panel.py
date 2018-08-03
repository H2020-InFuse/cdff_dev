import sys
import time
import warnings
import glob
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from . import logloader, typefromdict
from . import dataflowcontrol


class ControlPanelExpert:
    """Qt Application with EnviRe visualizer.

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

    def show_controls(self, stream_names, log_iterator, dfc):
        """Show control window to replay log file.

        Parameters
        ----------
        stream_names : list
            List of stream names available to be replayed

        log_iterator : generator
            Log data

        dfc : DataFlowControl
            Configured processing and data fusion logic
        """
        # splits args between image log data and other
        self.control_window = ControlPanelMainWindow(
            Step, self, self.stream_dict, stream_names, log_iterator, dfc)
        self.control_window.show()

    def exec_(self):
        """Start Qt application.

        Qt will take over the main thread until the main window is closed.
        """
        self.app.exec_()


class ControlPanelMainWindow(QMainWindow):
    """Instantiation of Qt window and Control Panel classes
    """

    def __init__(self, work, ctrl_pnl_expert, stream_dict, stream_names, log_iterator, dfc):
        super(ControlPanelMainWindow, self).__init__()
        self.worker = Worker(work, ctrl_pnl_expert, log_iterator, dfc)
        self.worker.start()
        QApplication.instance().aboutToQuit.connect(self.worker.quit)

        self.setWindowTitle("Visualization Control Panel")
        self.setMinimumSize(400, 310)

        central_widget = ControlPanelWidget(
            self.worker, ctrl_pnl_expert, stream_dict, stream_names)

        controller = ControlPanelController(
            central_widget, self.worker, ctrl_pnl_expert)

        central_widget.set_controller(controller)
        central_widget.add_widgets()
        self.setCentralWidget(central_widget)

    def closeEvent(self, QCloseEvent):
        self.worker.quit()


class ControlPanelWidget(QWidget):
    """Contains widgets that comprise physical layout
    of control panel

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
    """

    def __init__(self, worker, ctrl_pnl_expert, stream_dict, stream_names):
        super(ControlPanelWidget, self).__init__()
        self.stream_names = stream_names
        self.stream_dict = stream_dict
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
        self.step_info.setMaximum(2000000000)
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
        self.stream_edit.addItems(self.stream_names)
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
    """ Contains all functions for manipulating data and values 
    related to the visualization. These values are then passed to the
    ControlPanelExpert

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
            self.central_widget.data_edit.addItems(["Select data...", "Acceleration X", "Acceleration Y", "Acceleration Z",
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
        elif (data_choice not in self.ctrl_pnl_expert.data_types) and ("+" not in data_choice) and (data_choice != "No Data") and (data_choice != "Select data..."):
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


class Worker(QThread):
    """Worker thread that runs a loop in which callable is called.

    This thread is designed to be controlled precisely with a GUI.
    It is possible to do only step-wise execution of the work or introduce
    a break between steps. After each step the signal 'step_done' will be
    emitted. The current step count will be stored in the member 't_'.
    The length of the break is stored in 'break_length_'.

    Parameters
    ----------
    work : Callable
        The logic should be implemented in this callable. Note that this is
        a class that will be instantiated.

    worker_args : list
        Contructor paramters of work.

    worker_kwargs : dict
        Contructor paramters of work.
    """

    def __init__(self, work, ctrl_pnl_expert, log_iterator, dfc):
        self.work = work
        self.log_iterator = log_iterator
        self.dfc = dfc
        self.one_step = False
        self.all_steps = False
        self.keep_alive = True
        self.ctrl_pnl_expert = ctrl_pnl_expert

        self.break_length_ = 0.0
        self.t_ = 0

        super(Worker, self).__init__()

    def __del__(self):
        self.quit()

    step_done = pyqtSignal(int)

    def run(self):
        work = self.work(self.log_iterator, self.dfc)

        while self.keep_alive:
            try:
                if self.ctrl_pnl_expert.data_types:
                    if self.all_steps:
                        time.sleep(self.break_length_)
                        work()
                        self.t_ += 1
                        self.step_done.emit(self.t_)
                    elif self.one_step:
                        work()
                        self.t_ += 1
                        self.step_done.emit(self.t_)
                        self.one_step = False
                    else:
                        self.pause()
            except StopIteration:
                print("Reached the end of the logfile")
                break

    def step(self):
        self.one_step = True

    def play(self):
        print("playing")
        self.all_steps = True

    def pause(self):
        self.all_steps = False

    def quit(self):
        self.all_steps = False
        self.keep_alive = False


class Step:
    """A callable that replays one sample in each step."""

    def __init__(self, log_iterator, dfc):
        self.iterator = log_iterator
        self.dfc = dfc

    def __call__(self):
        timestamp, stream_name, typename, sample = next(self.iterator)
        # TODO: Better way to sort out usable data types
        if (typename != "int32_t") and (typename != "TransformerStatus") and (typename != "double") and (typename != "DepthMap") and (typename != "TimestampEstimatorStatus"):
            obj = typefromdict.create_from_dict(typename, sample)
            self.dfc.process_sample(
                timestamp=timestamp, stream_name=stream_name, sample=obj)
