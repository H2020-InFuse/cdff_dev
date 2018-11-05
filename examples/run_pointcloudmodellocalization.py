"""
==============================
Point Cloud Model Localisation
==============================

This DFPC will localise a pointcloud model in a larger point cloud.
The EnviRe visualizer will be started to display the two point clouds.
For this example, will will only search a point cloud in a slightly
translated version of it.
"""
import os
import sys
from PyQt4.QtGui import QApplication
from cdff_dev import path, envirevisualization, qtgui
from cdff_dev.extensions.pcl import helpers
from cdff_dev.dfpcs.pointcloudmodellocalisation import FeaturesMatching3D
import cdff_envire


def main():
    pc_original = helpers.load_ply_file(
        "test/test_data/pointclouds/dense_original.ply")
    pc_transformed = helpers.load_ply_file(
        "test/test_data/pointclouds/dense_transformed.ply")
    print(pc_transformed)
    print(pc_original)
    show_pointcloud([pc_original, pc_transformed])

    dfpc = FeaturesMatching3D()
    config_filename = os.path.join(
        path.load_cdffpath(),
        "Tests/ConfigurationFiles/DFPCs/PointCloudModelLocalisation/"
        "DfpcFeaturesMatching3D_conf01.yaml")
    dfpc.set_configuration_file(config_filename)
    dfpc.setup()

    dfpc.sceneInput(pc_transformed)
    dfpc.modelInput(pc_original)
    dfpc.computeModelFeaturesInput(True)
    dfpc.run()
    success = dfpc.successOutput()
    pose = dfpc.poseOutput()
    print(success)
    print(pose)


def show_pointcloud(pointclouds):
    class ShowPC:
        def __init__(self, graph, items, frame):
            self.graph = graph
            self.items = items
            self.frame = frame
            self.current_idx = 0

        def __call__(self):
            if self.current_idx >= len(self.items):
                return
            self.items[self.current_idx].add_to(self.graph, self.frame)
            self.current_idx += 1

    graph = cdff_envire.EnvireGraph()
    graph.add_frame("center")

    app = QApplication(sys.argv)

    vis = cdff_envire.EnvireVisualizer()
    vis.display_graph(graph, "center")
    vis.show()

    items = [envirevisualization.EnvireItem(pc) for pc in pointclouds]
    main_window = qtgui.ReplayMainWindow(ShowPC, graph, items, "center")
    main_window.show()

    app.exec_()
    del vis


if __name__ == "__main__":
    main()
