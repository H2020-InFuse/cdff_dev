from cdff_dev import dataflowcontrol, diagrams
from nose.tools import assert_in, assert_equal
import os
import tempfile


class LinearDFN:
    def __init__(self):
        self.w = 2.0
        self.b = 1.0
        self.x = 0.0

    def configure(self):
        pass

    def xInput(self, x):
        self.x = x

    def process(self):
        self.y = self.w * self.x + self.b

    def yOutput(self):
        return self.y


class SquareDFN:
    def __init__(self):
        self.x = 0.0

    def configure(self):
        pass

    def xInput(self, x):
        self.x = x

    def process(self):
        self.y = self.x ** 2

    def yOutput(self):
        return self.y


def test_graph_png():
    vis = dataflowcontrol.NoVisualization()
    nodes = {
        "linear": LinearDFN(),
        "square": SquareDFN()
    }
    periods = {
        "linear": 1,
        "square": 1
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x"),
        ("square.y", "result.y")
    )
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods, {}, vis)
    dfc.setup()
    filename = next(tempfile._get_candidate_names())
    try:
        diagrams.save_graph_png(dfc, filename)
    finally:
        if os.path.exists(filename):
            os.remove(filename)
