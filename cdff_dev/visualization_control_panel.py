import sys
import time
import warnings
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from . import logloader, typefromdict
from . import dataflowcontrol
import cdff_envire


class VisualizationController:
    """Qt Application with EnviRe visualizer.

    Parameters
    ----------
    frames : dict
        Mapping from port names to frame names

    urdf_files : list, optional (default: [])
        URDF files that should be loaded
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
        self.data_present = False
        self.type = None
        self.remove_outlier = False

    def show_controls(self, stream_names, log, dfc):
        """Show control window to replay log file.

        Parameters
        ----------
        stream_names : list
            List of stream names that should be replayed

        log : dict
            Log data

        dfc : DataFlowControl
            Configured processing and data fusion logic
        """
        self.control_window = ReplayMainWindow(
            Step, self, self.stream_dict, stream_names, log, dfc)
        self.control_window.show()

    def exec_(self):
        """Start Qt application.

        Qt will take over the main thread until the main window is closed.
        """
        self.app.exec_()

    def add_data_type(self, data_type):
        """Append  data to data_types list

        Parameters:
        ---------
        data_type: str
            Name of data type from stream
        """
        self.data_present = True
        self.data_types.append(data_type)

    def remove_data_type(self, data_type):
        """Remove data from data_types list

        Parameters:
        ---------
        data_type: str
            Name of data type from stream
        """
        self.data_types.remove(data_type)
        if not self.data_types:
            self.data_present = False

    def clear_data_types(self):
        """Set data_types list to empty and
        data_present to False
        """
        self.data_types = []
        self.data_present = False

    def set_outlier_file(self, file_):
        """Set outlier_file. If none entered,
        set to default file
        Parameters
        ----------
        file : str
            User entered file name
        """
        if file_ == "":
            self.outlier_file = "outliers.txt"
        else:
            self.outlier_file = file_
        print(self.outlier_file)


class ReplayMainWindow(QMainWindow):
    """Contains controls for replaying log files.
    """
    def __init__(self, work, vis_controller, stream_dict, *args, **kwargs):
        super(ReplayMainWindow, self).__init__()
        stream = args[0]
        self.worker = Worker(work, vis_controller, *args, **kwargs)
        self.worker.start()
        QApplication.instance().aboutToQuit.connect(self.worker.quit)

        self.setWindowTitle("Visualization Control Panel")
        self.setMinimumSize(400, 310)

        self.central_widget = ReplayControlWidget(
            self.worker, vis_controller, stream_dict, stream)
        self.setCentralWidget(self.central_widget)

    def closeEvent(self, QCloseEvent):
        self.worker.quit()


class ReplayControlWidget(QWidget):
    """Contains buttons and widgets, as well as
    the functions they connect to
    """

    def __init__(self, worker, vis_controller, stream_dict, stream):
        super(ReplayControlWidget, self).__init__()
        self.stream = stream
        self.stream_dict = stream_dict
        self.vis_controller = vis_controller
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
        self.step_button.clicked.connect(worker.step)
        self.step_button.setMinimumSize(200, 0)
        layout_sub_1.addWidget(self.step_button)

        # Play
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(worker.play)
        layout_sub_1.addWidget(self.play_button)

        # Pause
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(worker.pause)
        layout_sub_1.addWidget(self.pause_button)

        # Break label
        break_label = QLabel("Break between samples:")
        break_label.setAlignment(Qt.AlignBottom)
        layout_sub_1.addWidget(break_label)

        # Break between samples
        self.break_edit = QDoubleSpinBox()
        self.break_edit.setValue(worker.break_length_)
        self.break_edit.setMinimum(0.00)
        self.break_edit.setMaximum(sys.float_info.max)
        self.break_edit.setSingleStep(0.001)
        self.break_edit.setDecimals(3)
        self.break_edit.valueChanged.connect(self._update_break)
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

        # self.restart_replay = QPushButton("Restart Replay")
        # self.restart_replay.clicked.connect(self.restart_replay)
        # layout_sub_1.addWidget(self.restart_replay)

        # Data Selection label
        data_tools = QLabel("Data Selection")
        data_tools.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        layout_sub_2.addWidget(data_tools)

        # Stream select
        self.stream_edit = QComboBox()
        self.stream_edit.addItem("Select stream...")
        self.stream_edit.addItems(self.stream)
        self.stream_edit.currentIndexChanged.connect(self._set_stream)
        layout_sub_2.addWidget(self.stream_edit)

        # Data select
        self.data_edit = QComboBox()
        self.data_edit.addItems(["No Data"])
        layout_sub_2.addWidget(self.data_edit)

        h_box = QHBoxLayout()
        layout_sub_2.addLayout(h_box)

        # Add data
        self.data_add_button = QPushButton("Add Data")
        self.data_add_button.clicked.connect(self._add_data)
        h_box.addWidget(self.data_add_button)

        # Remove data
        self.data_rem_button = QPushButton("Remove Data")
        self.data_rem_button.clicked.connect(self._remove_data)
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
        self.xrange_edit.valueChanged.connect(self._update_max_xrange)
        h_box1.addWidget(self.xrange_edit)

        # Reset Y limits
        self.yrange_reset_button = QPushButton("Reset Y Limits")
        self.yrange_reset_button.clicked.connect(self._reset_yrange)
        h_box1.addWidget(self.yrange_reset_button)

        # Annotation tools label
        tools = QLabel("Annotation Tools")
        tools.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        layout_sub_2.addWidget(tools)

        # Outlier file select
        self.outlier_file_name = QLineEdit()
        self.outlier_file_name.setPlaceholderText("Save outliers to...")
        self.outlier_file_name.returnPressed.connect(self._set_outlier_file)
        layout_sub_2.addWidget(self.outlier_file_name)

        # Remove last selection
        self.outlier_remove_button = QPushButton("Undo last select")
        self.outlier_remove_button.clicked.connect(self._remove_outlier)
        layout_sub_2.addWidget(self.outlier_remove_button)

        self.worker = worker
        self.worker.step_done.connect(self.step_info.setValue)

    def _update_break(self):
        try:
            break_length = round(self.break_edit.value(),
                                 self.break_edit.decimals())
            if break_length < 0:
                raise ValueError("Length smaller than 0: %g" % break_length)
        except ValueError:
            print("Invalid break length: '%f'" % break_length)
        self.worker.break_length_ = break_length

    def _trigger_reset(self):
        self.vis_controller.stream_reset = True
        self.vis_controller.clear_data_types()
        self.data_edit.clear()

    def _set_stream(self):
        stream_choice = self.stream_edit.currentText()
        if stream_choice != self.vis_controller.stream and stream_choice != "Select stream...":
            self.vis_controller.stream = stream_choice
            for stream, type_ in self.stream_dict.items():
                if stream == stream_choice:
                    self._set_data_options(type_)
                    break

    def _set_data_options(self, type_):
        """Set available data options based on stream type
        """
        if type_ == "IMUSensors":
            self.vis_controller.type = "IMUSensors"
            self._trigger_reset()
            self.data_edit.addItems(["Select data...", "Acceleration X", "Acceleration Y", "Acceleration Z",
                                     "Gyroscopic X", "Gyroscopic Y", "Gyroscopic Z"])

        elif type_ == "Joints":
            self.vis_controller.type = "Joints"
            self._trigger_reset()
            self.data_edit.addItems(
                ["Select data...", "Position", "Effort", "Speed"])

        elif type_ == "LaserScan":
            self.vis_controller.type = "LaserScan"
            self._trigger_reset()
            self.data_edit.addItems(
                ["Select data...", "Speed", "Angular Resolution", "Start Angle"])

        elif type_ == "RigidBodyState":
            self.vis_controller.type = "RigidBodyState"
            self._trigger_reset()
            self.data_edit.addItems(
                ["Select data...", "Position X", "Position Y", "Position Z"])

        else:
            self._trigger_reset()
            self.data_edit.addItem("No Data")

    def _add_data(self):
        data_choice = self.data_edit.currentText()
        if len(self.vis_controller.data_types) == 4:
            print("Graph Display Max Reached")
        elif (data_choice not in self.vis_controller.data_types) and (data_choice != "No Data") and (data_choice != "Select data..."):
            self.vis_controller.add_data_type(data_choice)
            self.data_edit.setItemText(
                self.data_edit.currentIndex(), (data_choice + " +"))

    def _remove_data(self):
        data_choice = self.data_edit.currentText()[:-2]
        if (data_choice in self.vis_controller.data_types) and (data_choice != "Select data..."):
            self.vis_controller.remove_data_type(data_choice)
            self.data_edit.setItemText(
                self.data_edit.currentIndex(), data_choice)

    def _update_max_xrange(self):
        value = self.xrange_edit.value()
        try:
            max_xrange = float(value)
            if max_xrange < 0:
                raise ValueError("Width smaller than 0: %g" % max_xrange)
        except ValueError:
            print("Invalid viewport width: '%f'" % value)
        self.vis_controller.max_xrange = (int(value))

    def _reset_yrange(self):
        self.vis_controller.yrange_reset = True

    def _set_outlier_file(self):
        self.vis_controller.set_outlier_file(self.outlier_file_name.text())

    def _remove_outlier(self):
        self.vis_controller.remove_outlier = True



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

    def __init__(self, work, vis_controller, *worker_args, **worker_kwargs):
        self.work = work
        self.worker_args = worker_args
        self.worker_kwargs = worker_kwargs
        self.one_step = False
        self.all_steps = False
        self.keep_alive = True
        self.vis_controller = vis_controller

        self.break_length_ = 0.0
        self.t_ = 0

        super(Worker, self).__init__()

    def __del__(self):
        self.quit()

    step_done = pyqtSignal(int)

    def run(self):
        work = self.work(*self.worker_args, **self.worker_kwargs)

        while self.keep_alive:
            try:
                if self.vis_controller.data_present:
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
        self.all_steps = True

    def pause(self):
        self.all_steps = False

    def quit(self):
        self.all_steps = False
        self.keep_alive = False


class Step:
    """A callable that replays one sample in each step."""

    def __init__(self, stream_names, log, dfc):
        self.iterator = logloader.replay(stream_names, log, verbose=0)
        self.dfc = dfc

    def __call__(self):
        timestamp, stream_name, typename, sample = next(self.iterator)
        obj = typefromdict.create_from_dict(typename, sample)
        self.dfc.process_sample(
            timestamp=timestamp, stream_name=stream_name, sample=obj)
