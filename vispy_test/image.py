from PyQt4.QtGui import QWidget, QVBoxLayout, QApplication
import vispy
vispy.use("PyQt4")
import vispy.scene
import vispy.visuals


class ImageWidget(QWidget):
    def __init__(self, image_shape, parent=None):
        super(ImageWidget, self).__init__(parent)

        height, width = image_shape
        self.canvas = vispy.scene.SceneCanvas(
            parent=self, size=(width, height), bgcolor="lightgray")
        self.canvas.show()

        # Set up a viewbox to display the image with interactive pan/zoom
        view = self.canvas.central_widget.add_view()

        blank_image = np.empty(image_shape, dtype=np.uint8)
        if len(image_shape) == 2:
            blank_image[:, :] = 255
        else:
            blank_image[:, :, :] = 255
        self.image = vispy.scene.visuals.Image(
            blank_image, interpolation="nearest", parent=view.scene,
            method="subdivide", cmap="grays")

        # Set 2D camera (the camera will scale to the contents in the scene)
        view.camera = vispy.scene.PanZoomCamera(
            aspect=1, flip=(False, True, False))
        view.camera.set_range()

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.canvas.native)

    def write_data(self, port_name, sample, timestamp):
        self.image.set_data(sample)
        self.setWindowTitle(port_name)
        self.canvas.update()


import numpy as np

app = QApplication([])
#image_shape = (1032, 772)
image_shape = 833, 558
#image_shape = (600, 800)
image = (np.random.rand(*image_shape) * 255.0).astype(np.uint8)
image[:, -300:] = 255
image[:-300, :] = 255
w = ImageWidget(image_shape)
w.write_data("Test.port", image, None)
w.show()
app.exec_()