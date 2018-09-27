import sys
import numpy as np
from PyQt4.QtGui import QApplication
import pyqtgraph as pg
from . import dataflowcontrol, qtgui


# monkey-patching ImageItem.getHistogram for Python 3
def ImageItem_getHistogram(self, bins='auto', step='auto', targetImageSize=200,
                           targetHistogramSize=500, **kwds):
    """Returns x and y arrays containing the histogram values for the current image.
    For an explanation of the return format, see numpy.histogram().

    The *step* argument causes pixels to be skipped when computing the histogram to save time.
    If *step* is 'auto', then a step is chosen such that the analyzed data has
    dimensions roughly *targetImageSize* for each axis.

    The *bins* argument and any extra keyword arguments are passed to
    np.histogram(). If *bins* is 'auto', then a bin number is automatically
    chosen based on the image characteristics:

    * Integer images will have approximately *targetHistogramSize* bins,
      with each bin having an integer width.
    * All other types will have *targetHistogramSize* bins.

    This method is also used when automatically computing levels.
    """
    if self.image is None:
        return None, None
    if step == 'auto':
        # modification: cast to int
        step = (int(np.ceil(self.image.shape[0] / targetImageSize)),
                int(np.ceil(self.image.shape[1] / targetImageSize)))
    if np.isscalar(step):
        step = (step, step)
    stepData = self.image[::step[0], ::step[1]]

    if bins == 'auto':
        if stepData.dtype.kind in "ui":
            mn = stepData.min()
            mx = stepData.max()
            step = np.ceil((mx - mn) / 500.)
            bins = np.arange(mn, mx + 1.01 * step, step, dtype=np.int)
            if len(bins) == 0:
                bins = [mn, mx]
        else:
            bins = 500

    kwds['bins'] = bins
    hist = np.histogram(stepData, **kwds)

    return hist[1][:-1], hist[0]
pg.ImageItem.getHistogram = ImageItem_getHistogram


class ImageVisualizerApplication:
    """Qt Application with image visualizer.

    Parameters
    ----------
    stream_name : str
        Name of the stream that will be displayed
    """
    def __init__(self, stream_name):
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
        self.image_view_ = pg.ImageView(name=self.stream_name)
        self.image_view_.show()

    def report_node_output(self, port_name, sample, timestamp):
        if port_name == self.stream_name:
            image = sample.data.array_reference().copy()

            if image.ndim == 2:
                image = image.T
            elif image.ndim == 3:
                image = image.transpose(1, 0, 2)
            else:
                raise ValueError("Impossible number of channels: %d"
                                 % image.ndim)

            if sample.metadata.mode == "mode_GRAY":
                image = image.reshape(image.shape[0], image.shape[1])
            elif sample.metadata.mode not in [
                    "mode_RGB", "mode_RGBA", "mode_BGR", "mode_BGRA",
                    "mode_HSV", "mode_HLS", "mode_YUV", "mode_UYVY"]:
                raise ValueError("Don't know how to handle mode '%s'"
                                 % sample.metadata.mode)

            kwargs = dict()
            if sample.data.depth == "depth_8U":
                kwargs["autoLevels"] = False
                kwargs["levels"] = (0.0, 255.0)
            elif sample.data.depth == "depth_32F":
                kwargs["autoLevels"] = True
            else:
                raise ValueError("Unknown depth '%s'" % sample.data.depth)

            self.image_view_.setImage(image, **kwargs)
