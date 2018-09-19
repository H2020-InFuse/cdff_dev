from PyQt4.QtCore import QObject, pyqtSignal
from nose.tools import assert_equal, assert_greater
from time import sleep
from cdff_dev import qtgui


counter = 0
def test_worker():
    class Work(QObject):
        def __init__(self):
            super(Work, self).__init__()
            self.i = 0

        def __call__(self):
            global counter
            self.i += 1
            counter = self.i
            if self.i > 10000:
                raise StopIteration()

    class Emitter(QObject):
        step = pyqtSignal()
        play = pyqtSignal()
        pause = pyqtSignal()

    thread = qtgui.Worker(Work)
    thread.start()

    em = Emitter()
    em.step.connect(thread.step)
    em.play.connect(thread.play)
    em.pause.connect(thread.pause)

    assert_equal(counter, 0)
    em.step.emit()
    sleep(0.03)
    assert_equal(counter, 1)

    em.play.emit()
    sleep(0.01)
    em.pause.emit()
    sleep(0.03)
    old_counter = counter
    assert_greater(old_counter, 1)
    sleep(0.03)
    assert_equal(counter, old_counter)