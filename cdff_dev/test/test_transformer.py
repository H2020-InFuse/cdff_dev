from cdff_dev import transformer
import cdff_envire
import cdff_types
import numpy as np
from nose.tools import assert_equal, assert_true, assert_almost_equal


class Transformer(transformer.EnvireDFN):
    def __init__(self):
        super(Transformer, self).__init__()
        self.graph_ = cdff_envire.EnvireGraph()

    def initialize_graph(self, graph):
        graph.add_frame("B")
        graph.add_frame("C")
        graph.add_frame("D")
        graph.add_frame("E")

        t = cdff_envire.Transform()
        t.transform.translation.fromarray(np.array([4, 3, 4], dtype=np.float))
        graph.add_transform("D", "E", t)

    def A2BInput(self, data):
        self._set_transform(data)

    def B2CInput(self, data):
        self._set_transform(data)

    def C2DInput(self, data):
        self._set_transform(data)

    def D2AOutput(self):
        return self._get_transform("D", "A")

    def D2EOutput(self):
        return self._get_transform("D", "E")


def test_add_get_transformations():
    transformer = Transformer()

    A2B = cdff_types.RigidBodyState()
    A2B.source_frame = "A"
    A2B.target_frame = "B"
    A2B.pos.fromarray(np.array([1, 2, 3], dtype=np.float))
    transformer.A2BInput(A2B)

    B2C = cdff_types.RigidBodyState()
    B2C.source_frame = "B"
    B2C.target_frame = "C"
    B2C.pos.fromarray(np.array([3, 2, 1], dtype=np.float))
    transformer.B2CInput(B2C)

    C2D = cdff_types.RigidBodyState()
    C2D.source_frame = "C"
    C2D.target_frame = "D"
    C2D.pos.fromarray(np.array([5, 5, 5], dtype=np.float))
    transformer.C2DInput(C2D)

    D2A = transformer.D2AOutput()
    assert_equal(D2A.source_frame, "D")
    assert_equal(D2A.target_frame, "A")
    assert_equal(D2A.pos[0], -9)
    assert_equal(D2A.pos[1], -9)
    assert_equal(D2A.pos[2], -9)


def test_static_transformations():
    transformer = Transformer()

    D2E = transformer.D2EOutput()
    assert_equal(D2E.source_frame, "D")
    assert_equal(D2E.target_frame, "E")
    assert_equal(D2E.pos[0], 4)
    assert_equal(D2E.pos[1], 3)
    assert_equal(D2E.pos[2], 4)


def test_config():
    transformer = Transformer()
    transformer.set_configuration_file("test/test_data/transformer_config.msg")
    transformer.configure()

    graph = transformer.graph_

    assert_true(graph.contains_frame("config_sherpaTT_body"))
    assert_true(graph.contains_frame("config_imu"))
    assert_true(graph.contains_frame("config_tcp_base"))
    assert_true(graph.contains_frame("config_tcp"))
    assert_true(graph.contains_frame("config_camera_left"))
    assert_true(graph.contains_frame("config_camera_color"))
    assert_true(graph.contains_frame("config_velodyne"))

    assert_true(graph.contains_edge("config_sherpaTT_body", "config_imu"))
    assert_true(graph.contains_edge("config_imu", "config_tcp_base"))
    assert_true(graph.contains_edge("config_tcp_base", "config_tcp"))
    assert_true(graph.contains_edge("config_tcp", "config_camera_left"))
    assert_true(graph.contains_edge("config_tcp", "config_camera_color"))

    t = graph.get_transform("config_sherpaTT_body", "config_camera_left")
    assert_almost_equal(t.transform.translation[0], -0.045)
    assert_almost_equal(t.transform.translation[1], 0.62)
    assert_almost_equal(t.transform.translation[2], 0.2621)
    assert_equal(t.transform.orientation.toarray()[0], 0.85782960320932)
    assert_equal(t.transform.orientation.toarray()[1], 0.006085361255164068)
    assert_equal(t.transform.orientation.toarray()[2], -0.009559094197543583)
    assert_equal(t.transform.orientation.toarray()[3], -0.5138092680696382)


def test_replace_graph():
    transformer = Transformer()
    transformer.set_configuration_file("test/test_data/transformer_config.msg")
    transformer.configure()

    transformer.graph_ = cdff_envire.EnvireGraph()

    # static transformations should be set automatically
    graph = transformer.graph_

    assert_true(graph.contains_frame("B"))
    assert_true(graph.contains_frame("C"))
    assert_true(graph.contains_frame("D"))
    assert_true(graph.contains_frame("E"))

    assert_true(graph.contains_edge("D", "E"))

    assert_true(graph.contains_frame("config_sherpaTT_body"))
    assert_true(graph.contains_frame("config_imu"))
    assert_true(graph.contains_frame("config_tcp_base"))
    assert_true(graph.contains_frame("config_tcp"))
    assert_true(graph.contains_frame("config_camera_left"))
    assert_true(graph.contains_frame("config_camera_color"))
    assert_true(graph.contains_frame("config_velodyne"))

    assert_true(graph.contains_edge("config_sherpaTT_body", "config_imu"))
    assert_true(graph.contains_edge("config_imu", "config_tcp_base"))
    assert_true(graph.contains_edge("config_tcp_base", "config_tcp"))
    assert_true(graph.contains_edge("config_tcp", "config_camera_left"))
    assert_true(graph.contains_edge("config_tcp", "config_camera_color"))

    t = graph.get_transform("config_sherpaTT_body", "config_camera_left")
    assert_almost_equal(t.transform.translation[0], -0.045)
    assert_almost_equal(t.transform.translation[1], 0.62)
    assert_almost_equal(t.transform.translation[2], 0.2621)
    assert_equal(t.transform.orientation.toarray()[0], 0.85782960320932)
    assert_equal(t.transform.orientation.toarray()[1], 0.006085361255164068)
    assert_equal(t.transform.orientation.toarray()[2], -0.009559094197543583)
    assert_equal(t.transform.orientation.toarray()[3], -0.5138092680696382)
