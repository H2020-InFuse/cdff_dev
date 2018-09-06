import sys
from PyQt4.QtGui import *
import cdff_envire


def main():
    graph = cdff_envire.EnvireGraph()
    #graph.add_frame("center")

    #cdff_envire.load_urdf(graph,"./sherpa_tt-master/urdf/sherpa.urdf",False,False,True);
    cdff_envire.load_urdf(graph,"./bundle-sherpa_tt/data/urdf_sherpa_meshes_LQ/sherpa_tt.urdf",False,False,True);
    #cdff_envire.load_smurf(graph,"./sherpa_tt-master/smurf/sherpa.smurf",False,False,True);
    

    app = QApplication(sys.argv)



    vis = cdff_envire.EnvireVisualizer()
    #vis.display_graph(graph, "sherpa_tt")
    vis.display_graph(graph, "body")
    
    vis.show()



    app.exec_()
    del vis


if __name__ == "__main__":
    main()
