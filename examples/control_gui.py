import sys
from PyQt4.QtGui import *
from cdff_dev import envirevisualization


if __name__ == "__main__":
    class Work:
        def __init__(self):
            self.i = 0

        def __call__(self):
            print(self.i)
            self.i += 1
            if self.i > 10000:
                raise StopIteration()

    app = QApplication(sys.argv)
    win = envirevisualization.ReplayMainWindow(Work)
    win.show()
    app.exec_()
