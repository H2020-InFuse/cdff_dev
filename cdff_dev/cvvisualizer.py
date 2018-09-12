import cv2
from . import dataflowcontrol


class CVVisualizer(dataflowcontrol.VisualizationBase):
    def __init__(self, port_name):
        self.port_name = port_name

    def __del__(self):
        cv2.destroyAllWindows()

    def report_node_output(self, port_name, sample, timestamp):
        if port_name == self.port_name:
            cv2.imshow(port_name, sample.array_reference())
            cv2.waitKey(0)