import time
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *


class ReplayMainWindow(QMainWindow):
    def __init__(self, worker, parent=None):
        super(ReplayMainWindow, self).__init__(parent)

        self.worker = worker

        self.setWindowTitle("Log Replay")

        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        self.central_widget = ReplayControlWidget(self.worker)
        self.setCentralWidget(self.central_widget)


class ReplayControlWidget(QWidget):
    def __init__(self, worker):
        super(ReplayControlWidget, self).__init__()

        layout = QHBoxLayout()
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
        self.break_edit = QDoubleSpinBox()
        self.break_edit.setValue(worker.break_length)
        self.break_edit.setMinimum(0.0)
        self.break_edit.setMaximum(sys.float_info.max)
        self.break_edit.setSingleStep(0.01)

        layout.addWidget(self.break_edit)

        self.break_update_button = QPushButton("Update Break")
        self.break_update_button.clicked.connect(self.update_break)
        layout.addWidget(self.break_update_button)

        self.worker = worker

    def update_break(self):
        text = self.break_edit.text()
        try:
            break_length = float(text)
            if break_length < 0:
                raise ValueError("Length smaller than 0: %g" % break_length)
        except ValueError as e:
            print("Invalid break length: '%s'" % text)
        self.worker.break_length = break_length


class Worker(QThread):
    def __init__(self, work):
        self.work = work
        self.one_step = False
        self.all_steps = False
        self.keep_alive = True
        self.break_length = 0.0
        super(Worker, self).__init__()

    def __del__(self):
        self.quit()

    def run(self):
        work = self.work()

        # TODO would be better to signal that the main window is loaded
        #for _ in range(4):  # wait until widget is loaded
        #    time.sleep(1)
        #    if not self.keep_alive:
        #        break

        while self.keep_alive:
            try:
                if self.all_steps:
                    print(self.break_length)
                    time.sleep(self.break_length)
                    next(work)
                elif self.one_step:
                    next(work)
                    self.one_step = False
            except StopIteration:
                print("Reached the end of the logfile")
                break

        # TODO remove
        #while self.keep_alive:  # thread is kept alive to avoid a segfault
        #    time.sleep(1)

    def step(self):
        self.one_step = True

    def play(self):
        self.all_steps = True

    def pause(self):
        self.all_steps = False


    def quit(self):
        self.all_steps = False
        self.keep_alive = False
        time.sleep(1)


if __name__ == "__main__":
    def dummy_worker():
        for i in range(10000):
            print(i)
            yield

    app = QApplication(sys.argv)
    worker = Worker(dummy_worker)
    worker.start()
    win = ReplayMainWindow(worker)
    win.show()
    app.aboutToQuit.connect(worker.quit)
    app.exec_()
