import sys
from PyQt4.QtGui import *
import cdff_envire
import cdff_types


def main():
    graph = cdff_envire.EnvireGraph()
    graph.add_frame("center")

    graph.add_frame("map")
    trans = cdff_envire.TransformWithCovariance()
    trans.translation.x = 0
    trans.translation.y = 0
    trans.translation.z = 0
    trans.orientation = cdff_envire.Quaterniond()
    wrap = cdff_envire.Transform()
    wrap.transform = trans
    graph.add_transform("center", "map", wrap)

    m = cdff_types.Map()
    m.metadata.scale = 0.1
    m.data.rows = 100
    m.data.cols = 100
    m.data.channels = 1
    m.data.row_size = 0
    m.data.depth = "depth_32F"
    data = m.data.array_reference()

    for x in range(m.data.rows):
        for y in range(m.data.cols):
            data[x, y, 0] = (x + y) / 100.0

    item = cdff_envire.GenericItem()
    graph.add_item_to_frame("map", item, m)

    pose = cdff_types.RigidBodyState()
    poseitem = cdff_envire.GenericItem()
    graph.add_item_to_frame("center", poseitem, pose)

    app = QApplication(sys.argv)

    vis = cdff_envire.EnvireVisualizer()
    vis.display_graph(graph, "center")

    vis.show()

    app.exec_()
    del vis


if __name__ == "__main__":
    main()
