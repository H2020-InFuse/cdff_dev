"""
==============================
Point Cloud Model Localisation
==============================

This DFPC will localise a pointcloud model in a larger pointcloud.
We will download an external PLY file from the internet for this example.
We also need one additional Python module that is not installed with CDFF-Dev.
You can install it with

    sudo pip3 install plyfile
"""
import os
import sys
import shutil
import urllib.request
import tarfile
import plyfile
from PyQt4.QtGui import QApplication
from cdff_dev import path, envirevisualization, qtgui
from cdff_dev.dfpcs.pointcloudmodellocalisation import FeaturesMatching3D
import cdff_types
import cdff_envire


def main():
    #download_bunny()
    #original_ply = "test/test_data/bun_zipper_original.ply"
    #transformed_ply = "test/test_data/bun_zipper_transformed.ply"
    original_ply = "test/test_data/dense_original.ply"
    transformed_ply = "test/test_data/dense_transformed.ply"
    pc_full = pointcloud_from_ply(original_ply)
    pc_model = pointcloud_from_ply(transformed_ply)
    print(pc_model)
    print(pc_full)
    #show_pointcloud([pc_full, pc_model])

    dfpc = FeaturesMatching3D()
    config_filename = os.path.join(
        path.load_cdffpath(),
        "Tests/ConfigurationFiles/DFPCs/PointCloudModelLocalisation/"
        "DfpcFeaturesMatching3D_conf01.yaml")
    print(config_filename)
    dfpc.set_configuration_file(config_filename)
    dfpc.setup()

    dfpc.sceneInput(pc_full)
    dfpc.modelInput(pc_model)
    dfpc.computeModelFeaturesInput(True)
    print("Running...")
    sys.stdout.flush()
    dfpc.run()
    print("DONE")
    pose = dfpc.poseOutput()
    print(pose)


def download_bunny(verbose=1):
    filename = "test/test_data/bunny.tar.gz"
    if os.path.exists(filename):
        if verbose:
            print("Found bunny, not downloading.")
        return

    if verbose:
        print("Downloading bunny... ", end="")
        sys.stdout.flush()

    url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
    with urllib.request.urlopen(url) as response, open(filename, "wb") as f:
        shutil.copyfileobj(response, f)

    if verbose:
        print("Uncompressing... ", end="")
        sys.stdout.flush()

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path="test/test_data/")

    # from CDFF/build/Tests/DataGenerators/:
    # echo "remove_outliers save ../../../../CDFF_dev/test/test_data/bun_zipper_original.ply quit" | ./point_cloud_transformer ../../../../CDFF_dev/test/test_data/bunny/reconstruction/bun_zipper.ply
    # echo "remove_outliers transform 0.1 0 0 0 0 0 1 save ../../../../CDFF_dev/test/test_data/bun_zipper_transformed.ply quit" | ./point_cloud_transformer ../../../../CDFF_dev/test/test_data/bunny/reconstruction/bun_zipper.ply

    if verbose:
        print("DONE")


def pointcloud_from_ply(filename):
    plydata = plyfile.PlyData.read(filename)
    vertices = plydata.elements[0]
    pc = cdff_types.Pointcloud()
    pc.data.points.resize(vertices.count)
    for i in range(vertices.count):
        for j in range(3):
            pc.data.points[i, j] = vertices.data[i][j]
    return pc


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
            print("Adding item to graph...", end="")
            sys.stdout.flush()
            self.items[self.current_idx].add_to(self.graph, self.frame)
            print("DONE")
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
