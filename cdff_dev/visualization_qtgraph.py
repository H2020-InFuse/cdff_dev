import sys
import numpy as np
from PyQt4.QtGui import QApplication
from PyQt4.QtGui import QImage
from PyQt4.QtGui import QLabel
from PyQt4.QtGui import QPixmap
import PyQt4.QtCore as QtCore
from PyQt4.QtCore import Qt
#import pyqtgraph as pg
from . import dataflowcontrol, qtgui
import cv2
import pprint
import cdff_envire
from PIL import Image

#http://doc.qt.io/qt-5/qtwidgets-widgets-imageviewer-example.html#

class ImageVisualizerApplication:
    """Qt Application with image visualizer.

    Parameters
    ----------
    stream_name : str
        Name of the stream that will be displayed
    """

    def __init__(self, stream_name):
        cdff_envire.x_init_threads()
        self.app = QApplication(sys.argv)
        self.stream_name = stream_name
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
        self.visualization = ImageVisualization(self.stream_name)
        dfc.register_visualization(self.visualization)
        self.control_window.show()

    def exec_(self):
        """Start Qt application.

        Qt will take over the main thread until the main window is closed.
        """
        self.app.exec_()


# TODO dependencies: python3-pyqtgraph (python3-pyqt4.qtopengl for 3D)
class ImageVisualization(dataflowcontrol.VisualizationBase):
    def __init__(self, stream_name):
        self.stream_name = stream_name
        self.image = QImage()
        self.image_view_ = QLabel("test")
        #self.image_view_.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)                                                                                                                                   
        self.image_view_.resize(640,480)                                                                                                                                 
        #self.image_view_.setAlignment(QtCore.AlignCenter)   

        #self.image_view_.setPixmap(QPixmap.fromImage(self.image))
        self.image_view_.show()
        #self.image_lock = threading.Lock()
        #self.image_view_.moveToThread(QApplication.instance().thread())

    def report_node_output(self, port_name, sample, timestamp):
        if port_name == self.stream_name:
            imagecopy = sample.data.array_reference().copy()


            #img = cv2.Mat(sample.data.rows,sample.data.cols,cv2.CV_8UC1)
            #img.data = image.__array_interface__['data']
            
            rgbimage = cv2.cvtColor(imagecopy, cv2.COLOR_GRAY2RGB)
            resized_image = cv2.resize(rgbimage, (320, 240)) 
            #bytergbimage = cv2.convertTo(resized_image,cv2.CV_8UC3)
            cv2.imshow('image',resized_image)

            print (resized_image.dtype)
            
            #img = Image.fromarray(resized_image, 'RGB')
            #img.show()

            #cv2.waitKey(1)

            # if image.ndim == 2:
            #     image = image.T
            # elif image.ndim == 3:
            #     image = image.transpose(1, 0, 2).copy()
            # else:
            #     raise ValueError("Impossible number of channels: %d"
            #                      % image.ndim)

            # if sample.metadata.mode == "mode_GRAY":
            #     #rgbimage = np.dstack(image,image)
            #     #rgbimage = np.dstack(rgbimage,image)
            #     image = image.reshape(image.shape[0], image.shape[1])
            #     #format = QImage.Format_Grayscale8
            # elif sample.metadata.mode not in [
            #         "mode_RGB", "mode_RGBA", "mode_BGR", "mode_BGRA",
            #         "mode_HSV", "mode_HLS", "mode_YUV", "mode_UYVY"]:
            #     raise ValueError("Don't know how to handle mode '%s'"
            #                      % sample.metadata.mode)

            # # if sample.data.channels == 3

            # # if sample.data.depth == "depth_8U":
                
            # # elif sample.data.depth == "depth_32F":

            # # else:
            # #     raise ValueError("Unknown depth '%s'" % sample.data.depth)
            # self.image_view_ = QLabel("test")
            # self.image_view_.resize(640,400)   
            # #https://stackoverflow.com/questions/48639185/pyqt5-qimage-from-numpy-array
            #self.image = QImage(rgbimage,rgbimage.shape[1],rgbimage.shape[0],format)

            self.image = QImage(rgbimage,sample.data.rows,sample.data.cols,QImage.Format_RGB888)
            #self.image.load("test.png")
            #imageWindowGlob.setPixmap(QPixmap.fromImage(self.image))
            #self.image_view_.setPixmap(QPixmap.fromImage(self.image))
            #self.image_view_.moveToThread(QApplication.instance().thread())
            #self.image_view_.emit(QtCore.SIGNAL('setPixmap(QImage)'), self.image)
            pixmap = QPixmap.fromImage(self.image)
            pixmap = pixmap.scaled(320, 240, Qt.KeepAspectRatio)
            #pixmap = pixmap.scaled(640,400, Qt.KeepAspectRatio)
            QtCore.QMetaObject.invokeMethod( self.image_view_, "setPixmap", QtCore.Q_ARG( QPixmap , pixmap ) )

            self.image_view_.show()
            

            #self.image_view_.setImage(image, **kwargs)
