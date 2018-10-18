import sys
from PyQt4.QtGui import QApplication
import cdff_envire
import _thread, time


def set_value(model):
    val=0
    while True:
        model.set_joint_angle("gamma_rear_left", val)
        val+=1
        time.sleep(1)

def main():
    graph = cdff_envire.EnvireGraph()
    urdfModel= cdff_envire.EnvireURDFModel()
    urdfModel.load_urdf(
        graph, "./bundle-sherpa_tt/data/urdf_sherpa_meshes_LQ/sherpa_tt.urdf",
        load_visuals=True)

    app = QApplication(sys.argv)

    vis = cdff_envire.EnvireVisualizer()

    res =  urdfModel.set_joint_angle("phi_rear_left", 1)
    #res =  urdfModel.set_joint_angle("gamma_rear_left", 90)

    _thread.start_new_thread( set_value, (urdfModel,) )

    vis.display_graph(graph, "body")
    vis.show()




    app.exec_()
    del vis


if __name__ == "__main__":
    main()
