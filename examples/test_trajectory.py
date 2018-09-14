from cdff_dev import envirevisualization, dataflowcontrol
from PyQt4.QtGui import *
import numpy as np
import time


def data_generator():
    for i in np.arange(100):
        sample = [i,i,i]
        typename = 'Vector3d'
        timestamp = i
        stream_name = 'data.point'
        yield timestamp, stream_name, typename, sample


def main():
    sample_iter = data_generator()

    frames = {"data.point": "center"}
    urdf_files=[]
    app = envirevisualization.EnvireVisualizerApplication(
        frames, urdf_files, center_frame="center")

    connections = (('data.point', 'result.trajectory'),)
    dfc = dataflowcontrol.DataFlowControl(
        {}, connections, {}, real_time=False)
    dfc.setup()

    app.show_controls(sample_iter, dfc)

    app.exec_()
    del vis


if __name__ == "__main__":
    main()
