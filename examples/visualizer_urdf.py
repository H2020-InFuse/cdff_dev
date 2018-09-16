import sys
from PyQt4.QtGui import QApplication
import cdff_envire


def main():
    graph = cdff_envire.EnvireGraph()
    cdff_envire.load_urdf(
        graph, "test/test_data/urdf_sherpa_meshes_LQ/sherpa_tt.urdf",
        load_visuals=True)

    app = QApplication(sys.argv)

    vis = cdff_envire.EnvireVisualizer()
    vis.display_graph(graph, "body")
    vis.show()

    app.exec_()
    del vis


if __name__ == "__main__":
    main()
