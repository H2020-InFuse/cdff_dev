import sys
from PyQt4.QtGui import *
import cdff_envire
import cdff_types
import time
from threading import Thread
import numpy as np


def map_update_thread (map,item):
    while True:
        # print("update")
        item.set_data(map)
        time.sleep(1)

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
    wrap._set_transform(trans)
    graph.add_transform("center","map",wrap)

    map = cdff_types.Map()

    map.metadata.scale = 0.1
    map.data.rows = 100
    map.data.cols = 100
    map.data.channels = 1
    map.data.row_size = 0
    map.data._set_depth("depth_32F")

    data = map.data.array_reference()

    for x in range(0,map.data.rows):
        for y in range(0,map.data.cols):
            data[x,y,0] = (x+y)/100

    item = cdff_envire.GenericItem()
    #graph.add_item_to_frame("map", item, map)

    
    
    

    pose = cdff_types.RigidBodyState()
    poseitem = cdff_envire.GenericItem()
    graph.add_item_to_frame("center", poseitem, pose)

    graph.save_to_file("envire_graph.bin")

    graph2 = cdff_envire.EnvireGraph()
    graph2.load_from_file("envire_graph.bin")


    # item = EnvireItem(map)
    # item.add_to("center")

    app = QApplication(sys.argv)

    vis = cdff_envire.EnvireVisualizer()
    vis.display_graph(graph2, "center")


    # t = Thread(target=map_update_thread, args=(map,item,))
    # t.start()


    vis.show()

    

    app.exec_()
    del vis


if __name__ == "__main__":
    main()
