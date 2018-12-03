import sys
import time
from PyQt4.QtGui import QApplication
from cdff_dev.qtgui import Worker
import cdff_envire



class Step:
    def __init__(self, model):
        self.model = model
        self.angle = 0.0
        self.positive = True

    def __call__(self):
        #height
        self.model.set_joint_angle("gamma_front_left", self.angle)
        self.model.set_joint_angle("gamma1_fake_front_left", -self.angle)
        self.model.set_joint_angle("gamma2_fake_front_left", self.angle)
        self.model.set_joint_angle("beta_front_left", -self.angle)
        self.model.set_joint_angle("beta1_fake_front_left", self.angle)
        self.model.set_joint_angle("beta2_fake_front_left", -self.angle)

        #steer
        self.model.set_joint_angle("phi_front_left", self.angle)

        #wheel
        self.model.set_joint_angle("omega_front_left", self.angle)
        
        if self.positive:
            self.angle += 0.01
            if self.angle >= 1.57:
                self.positive = False
        else:
            self.angle -= 0.01
            if self.angle <= -1.57:
                self.positive = True

        time.sleep(0.001)


def main():
    graph = cdff_envire.EnvireGraph()
    urdf_model = cdff_envire.EnvireURDFModel()
    urdf_model.load_urdf(
        graph,
        "test/test_data/urdf_sherpa_meshes_LQ/sherpa_tt.urdf",
        load_visuals=True)

    app = QApplication(sys.argv)

    vis = cdff_envire.EnvireVisualizer()

    worker = Worker(Step, urdf_model)
    worker.started.connect(worker.play)
    worker.start()
    app.aboutToQuit.connect(worker.quit)

    vis.display_graph(graph, "body")
    vis.show()

    app.exec_()
    del vis


if __name__ == "__main__":
    main()
