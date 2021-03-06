import sys
import numpy as np
from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import SLOT
from PyQt4.QtCore import QMutex

from PyQt4.QtGui import QApplication
from PyQt4.QtGui import QImage
from PyQt4.QtGui import QPainter
from PyQt4.QtGui import QWidget
from PyQt4.QtGui import QHBoxLayout

from . import dataflowcontrol, qtgui
import cv2


class ImageVisualizerApplication:
    """Qt Application with image visualizer.

    Parameters
    ----------
    stream_name : str
        Name of the stream that will be displayed

    value_range : pair, optional (default: None)
        Lower and upper boundaries of values stored in pixels
    """
    def __init__(self, stream_name, value_range=None):
        self.app = QApplication(sys.argv)
        self.stream_name = stream_name
        self.value_range = value_range
        self.control_window = None

    def show_controls(self, iterator, dfc):
        """Show control window to replay log file.

        Parameters
        ----------
        iterator : Iterable
            Iterable object that yields log samples in the correct temporal
            order. The iterable returns in each step a quadrupel of
            (timestamp, stream_name, typename, sample).

        dfc : DataFlowControl
            Configured processing and data fusion logic
        """
        self.control_window = qtgui.ReplayMainWindow(qtgui.Step, iterator, dfc)
        self.visualization = ImageVisualization(
            self.stream_name, self.value_range)
        dfc.register_visualization(self.visualization)
        self.control_window.show()

    def exec_(self):
        """Start Qt application.

        Qt will take over the main thread until the main window is closed.
        """
        self.app.exec_()


class ImageVisualization(dataflowcontrol.VisualizationBase):
    def __init__(self, stream_name, value_range=None):
        self.stream_name = stream_name
        self.value_range = value_range
        self.image = QImage("main")
        self.image_widget = ImageWidget()
        self.image_widget.setWindowTitle(stream_name)
        self.image_widget.show()


    def report_node_output(self, port_name, sample, timestamp):
        if port_name == self.stream_name:
            image = _convert_to_uint8_rgb(sample, self.value_range)
            self.image = QImage(
                image, sample.data.cols, sample.data.rows,
                QImage.Format_RGB888)
            self.image_widget.setImage(self.image)


#http://doc.qt.io/qt-5/qtwidgets-widgets-imageviewer-example.html
#https://stackoverflow.com/questions/10307245/how-to-display-the-frames-of-a-video-via-qt-gui-application
#https://stackoverflow.com/questions/1242005/what-is-the-most-efficient-way-to-display-decoded-video-frames-in-qt/2671834#2671834
#https://stackoverflow.com/questions/33201384/pyqt-opengl-drawing-simple-scenes
#https://doc.qt.io/archives/qq/qq26-pyqtdesigner.html
class ImageWidget(QWidget):
    __pyqtSignals__ = ("imageUpdated()")

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.connect(self, SIGNAL("imageUpdated()"), self, SLOT("update()"))
        self.image = QImage()
        self.mutex = QMutex()
        self._size_initialized = False

    #overload paint event
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, 1)
        self.mutex.lock()
        painter.drawImage(self.rect(), self.image)
        if not self._size_initialized and self.image.width() > 0:
            self.setMinimumSize(
                self.image.width() // 4, self.image.height() // 4)
            self.resize(self.image.width() // 2, self.image.height() // 2)
            self._size_initialized = True
        self.mutex.unlock()

    #set image (thread safe)
    def setImage(self, newimage):
        self.mutex.lock()
        self.image = newimage
        self.mutex.unlock()
        #calling update via signal/slot (decouples threading)
        self.emit(SIGNAL("imageUpdated()"))


class ImagePairVisualizerApplication(ImageVisualizerApplication):
    """Qt Application with image pair visualizer.

    Parameters
    ----------
    stream_name : str
        Name of the stream that will be displayed

    value_range : pair, optional (default: None)
        Lower and upper boundaries of values stored in pixels
    """

    def show_controls(self, iterator, dfc):
        """Show control window to replay log file.

        Parameters
        ----------
        iterator : Iterable
            Iterable object that yields log samples in the correct temporal
            order. The iterable returns in each step a quadrupel of
            (timestamp, stream_name, typename, sample).

        dfc : DataFlowControl
            Configured processing and data fusion logic
        """
        self.control_window = qtgui.ReplayMainWindow(qtgui.Step, iterator, dfc)
        self.visualization = ImagePairVisualization(
            self.stream_name, self.value_range)
        dfc.register_visualization(self.visualization)
        self.control_window.show()


class ImagePairVisualization(dataflowcontrol.VisualizationBase):
    def __init__(self, stream_name, value_range=None):
        self.stream_name = stream_name
        self.value_range = value_range
        self.images = [QImage("left"), QImage("right")]
        self.widget = ImagePairWidget()
        self.widget.setWindowTitle(stream_name)
        self.widget.show()

    def report_node_output(self, port_name, sample, timestamp):
        if port_name == self.stream_name:
            for i, img in enumerate([sample.left, sample.right]):
                img = _convert_to_uint8_rgb(img, self.value_range)
                self.images[i] = QImage(
                    img, sample.left.data.cols, sample.left.data.rows,
                    QImage.Format_RGB888)
            self.widget.setImages(self.images)


class ImagePairWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        # to let the widgets resize themselves, we don't set the parent widget
        self.left_widget = ImageWidget()
        self.right_widget = ImageWidget()
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.left_widget)
        self.layout.addWidget(self.right_widget)
        self.setLayout(self.layout)
        self.resize(200, 100)

    def setImages(self, newimages):
        self.left_widget.setImage(newimages[0])
        self.right_widget.setImage(newimages[1])


def _convert_to_uint8_rgb(sample, value_range=None):
    image_ref = sample.data.array_reference()
    if value_range is not None:
        image = np.subtract(image_ref, value_range[0])
        r = value_range[1] - value_range[0]
        np.multiply(image, 255.0 / r, out=image)
        np.clip(image, 0.0, 255.0, out=image)
    else:
        image = image_ref
    image = image.astype(np.uint8, copy=True)

    if sample.metadata.mode=="mode_GRAY":
        rgbimage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif sample.metadata.mode=="mode_RGB":
        rgbimage = image
    elif sample.metadata.mode=="mode_RGBA":
        rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif sample.metadata.mode=="mode_BGR":
        rgbimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif sample.metadata.mode=="mode_BGRA":
        rgbimage = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif sample.metadata.mode=="mode_HSV":
        rgbimage = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    elif sample.metadata.mode=="mode_HLS":
        rgbimage = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    elif sample.metadata.mode=="mode_YUV":
        rgbimage = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    elif sample.metadata.mode not in ["mode_UYVY"]:
        raise ValueError("Don't know how to handle mode '%s'"
                         % sample.metadata.mode)

    return rgbimage