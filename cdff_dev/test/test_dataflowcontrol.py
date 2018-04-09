from cdff_dev import dataflowcontrol
from nose.tools import assert_in, assert_equal


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


def test_dfc():
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
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods, vis)
    dfc.setup()
    for i in range(101):
        dfc.process_sample(timestamp=i, stream_name="log.x", sample=i)
    dfc.process(timestamp=102)
    assert_in("linear.y", vis.data)
    assert_equal(vis.data["linear.y"][0], 201.0)
    assert_in("square.y", vis.data)
    assert_equal(vis.data["square.y"][0], 40401.0)