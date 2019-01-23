import sys
import time
from PyQt4.QtGui import (QApplication, QMainWindow, QMenuBar, QWidget,
                         QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
                         QVBoxLayout, QHBoxLayout)
from PyQt4.QtCore import QThread, pyqtSignal
from . import typefromdict


class ReplayMainWindow(QMainWindow):  # pragma: no cover
    """Contains controls for replaying log files."""
    def __init__(self, work, *args, **kwargs):
        super(ReplayMainWindow, self).__init__()

        self.worker = Worker(work, *args, **kwargs)
        self.worker.start()
        QApplication.instance().aboutToQuit.connect(self.worker.quit)

        self.setWindowTitle("Log Replay")

        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        self.central_widget_ = ReplayControlWidget(self.worker)
        self.setCentralWidget(self.central_widget_)

    def add_widget(self, widget):
        self.central_widget_.add_widget(widget)

    def closeEvent(self, QCloseEvent):
        self.worker.quit()


class ReplayControlWidget(QWidget):  # pragma: no cover
    """Contains buttons etc."""
    def __init__(self, worker):
        super(ReplayControlWidget, self).__init__()

        self.control_layout = QVBoxLayout()

        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(worker.step)
        self.control_layout.addWidget(self.step_button)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(worker.play)
        self.control_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(worker.pause)
        self.control_layout.addWidget(self.pause_button)

        self.control_layout.addWidget(QLabel("Break between samples:"))
        self.break_edit = QDoubleSpinBox()
        self.break_edit.setValue(worker.break_length_)
        self.break_edit.setMinimum(0.0)
        self.break_edit.setMaximum(sys.float_info.max)
        self.break_edit.setSingleStep(0.001)
        self.control_layout.addWidget(self.break_edit)

        self.break_update_button = QPushButton("Update Break")
        self.break_update_button.clicked.connect(self.update_break)
        self.control_layout.addWidget(self.break_update_button)

        self.control_layout.addWidget(QLabel("Current sample:"))
        self.step_info = QSpinBox()
        self.step_info.setMaximum(2000000000)
        self.step_info.setEnabled(False)
        self.control_layout.addWidget(self.step_info)

        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.control_layout)
        self.setLayout(self.main_layout)

        self.worker = worker
        self.worker.step_done.connect(self.step_info.setValue)

    def add_widget(self, widget):
        self.main_layout.addWidget(widget)

    def update_break(self):
        text = self.break_edit.text()
        # NOTE: workaround for German style numbers
        text = text.replace(",", ".")
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
        """Main loop."""
        # Note that the code coverage is not measured correctly for QThreads
        # in coveragepy: https://github.com/nedbat/coveragepy/issues/686
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
        """Execute only one step."""
        self.one_step = True

    def play(self):
        """Continue replay."""
        self.all_steps = True

    def pause(self):
        """Pause replay."""
        self.all_steps = False

    def quit(self):
        """Stop replay and main loop."""
        self.all_steps = False
        self.keep_alive = False


class Step:
    """A callable that replays one sample in each step."""
    def __init__(self, iterator, dfc):
        self.iterator = iterator
        self.dfc = dfc

    def __call__(self):
        timestamp, stream_name, typename, sample = next(self.iterator)
        obj = typefromdict.create_from_dict(typename, sample)
        self.dfc.process_sample(
            timestamp=timestamp, stream_name=stream_name, sample=obj)

