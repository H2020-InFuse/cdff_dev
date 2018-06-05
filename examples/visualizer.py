import sys
from PyQt4.QtGui import *
import cdff_envire


def main():
    graph = cdff_envire.EnvireGraph()
    graph.add_frame("center")

    app = QApplication(sys.argv)

    vis = cdff_envire.EnvireVisualizer()
    vis.display_graph(graph, "center")
    vis.show()

    app.exec_()
    del vis


if __name__ == "__main__":
    main()
