import sys
import time
import warnings
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from . import logloader, typefromdict
from . import dataflowcontrol
import cdff_envire


class EnvireVisualizerApplication:
    """Qt Application with EnviRe visualizer.

    Parameters
    ----------
    frames : dict
        Mapping from port names to frame names

    urdf_files : list, optional (default: [])
        URDF files that should be loaded
    """
    def __init__(self, frames, urdf_files=[]):
        self.app = QApplication(sys.argv)
        self.visualization = EnvireVisualization(frames, urdf_files)
        self.control_window = None

    def __del__(self):
        # Make sure to remove all items before visualizer is deleted,
        # otherwise the published events will result in a segfault!
        self.visualization.world_state_.remove_all_items()
        # Visualizer must be deleted before graph, otherwise
        # unsubscribing from events will result in a segfault!
        del self.visualization.visualizer

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
        dfc.set_visualization(self.visualization)
        self.control_window = ReplayMainWindow(Step, stream_names, log, dfc)
        self.control_window.show()

    def exec_(self):
        self.app.exec_()


class EnvireVisualization(dataflowcontrol.VisualizationBase):
    """EnviRe visualization.

    Parameters
    ----------
    frames : dict
        Mapping from port names to frame names

    urdf_files : list, optional (default: [])
        URDF files that should be loaded
    """
    def __init__(self, frames, urdf_files=[]):
        self.world_state_ = WorldState(frames, urdf_files)
        self.visualizer = cdff_envire.EnvireVisualizer()
        center_frame = list(frames.values())[0]  # TODO let user define center frame
        self.visualizer.display_graph(self.world_state_.graph_, center_frame)
        self.visualizer.show()

    def report_node_output(self, port_name, sample, timestamp):
        self.world_state_.report_node_output(port_name, sample, timestamp)
        self.visualizer.redraw()


class WorldState:
    """Represents the estimated world state of the system based on log data.

    Parameters
    ----------
    frames : dict
        Mapping from port names to frame names

    urdf_files : list
        URDF files that should be loaded
    """
    def __init__(self, frames, urdf_files):
        self.frames = frames
        self.items = dict()
        self.samples = dict()
        self.graph_ = cdff_envire.EnvireGraph()
        for filename in urdf_files:
            cdff_envire.load_urdf(self.graph_, filename)
        for frame in self.frames.values():
            if not self.graph_.contains_frame(frame):
                self.graph_.add_frame(frame)
        self.items_initialized = True

    def __del__(self):
        if self.items_initialized:
            self.remove_all_items()
        del self.graph_

    def remove_all_items(self):
        """Remove all items from EnviRe graph.

        This has to be done manually before any attached visualizer is deleted.
        """
        for port_name in self.items.keys():
            self.graph_.remove_item_from_frame(
                self.items[port_name], self.samples[port_name])
        self.items_initialized = False

    def report_node_output(self, port_name, sample, timestamp):
        if port_name not in self.frames:
            warnings.warn("No frame registered for port '%s'" % port_name)
            return

        self.samples[port_name] = sample

        # TODO set time stamp
        if port_name in self.items:
            item = self.items[port_name]
            item.set_data(sample)
        else:
            item = cdff_envire.GenericItem()
            self.items[port_name] = item
            try:
                self.graph_.add_item_to_frame(
                    self.frames[port_name], item, sample)
            except TypeError as e:
                warnings.warn("Cannot store type '%s' in EnviRe graph. "
                              "Reason: %s" % (type(sample), e))
        item.set_time(sample, timestamp)


class ReplayMainWindow(QMainWindow):
    def __init__(self, work, *args, **kwargs):
        super(ReplayMainWindow, self).__init__()

        self.worker = Worker(work, *args, **kwargs)
        self.worker.start()
        QApplication.instance().aboutToQuit.connect(self.worker.quit)

        self.setWindowTitle("Log Replay")

        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        self.central_widget = ReplayControlWidget(self.worker)
        self.setCentralWidget(self.central_widget)

    def closeEvent(self, QCloseEvent):
        self.worker.quit()


class ReplayControlWidget(QWidget):
    def __init__(self, worker):
        super(ReplayControlWidget, self).__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(worker.step)
        layout.addWidget(self.step_button)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(worker.play)
        layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(worker.pause)
        layout.addWidget(self.pause_button)

        layout.addWidget(QLabel("Break between samples:"))
        self.break_edit = QDoubleSpinBox()
        self.break_edit.setValue(worker.break_length_)
        self.break_edit.setMinimum(0.0)
        self.break_edit.setMaximum(sys.float_info.max)
        self.break_edit.setSingleStep(0.01)
        layout.addWidget(self.break_edit)

        self.break_update_button = QPushButton("Update Break")
        self.break_update_button.clicked.connect(self.update_break)
        layout.addWidget(self.break_update_button)

        layout.addWidget(QLabel("Current sample:"))
        self.step_info = QSpinBox()
        self.step_info.setMaximum(2000000000)
        self.step_info.setEnabled(False)
        layout.addWidget(self.step_info)

        self.worker = worker
        self.worker.step_done.connect(self.step_info.setValue)

    def update_break(self):
        text = self.break_edit.text()
        try:
            break_length = float(text)
            if break_length < 0:
                raise ValueError("Length smaller than 0: %g" % break_length)
        except ValueError as e:
            print("Invalid break length: '%s'" % text)
        self.worker.break_length_ = break_length


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
    def __init__(self, work, *worker_args, **worker_kwargs):
        self.work = work
        self.worker_args = worker_args
        self.worker_kwargs = worker_kwargs
        self.one_step = False
        self.all_steps = False
        self.keep_alive = True

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
    def __init__(self, stream_names, log, dfc):
        self.iterator = logloader.replay(stream_names, log, verbose=0)
        self.dfc = dfc

    def __call__(self):
        timestamp, stream_name, typename, sample = next(self.iterator)
        obj = typefromdict.create_from_dict(typename, sample)
        self.dfc.process_sample(
            timestamp=timestamp, stream_name=stream_name, sample=obj)