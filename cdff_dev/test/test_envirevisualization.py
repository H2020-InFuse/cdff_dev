from cdff_dev import envirevisualization
from PyQt4.QtCore import QObject, pyqtSignal
from nose.tools import assert_equal, assert_greater, assert_true
from time import sleep


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

    thread = envirevisualization.Worker(Work)
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


def test_world_state():
    world_state = envirevisualization.WorldState(
        {"a": "test1", "b": "test2"},
        ["test/test_data/model.urdf"])
    assert_true(world_state.graph_.contains_frame("test1"))
    assert_true(world_state.graph_.contains_frame("body"))