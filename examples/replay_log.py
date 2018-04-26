from cdff_dev import dataflowcontrol
from cdff_dev import diagrams


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


def main():
    vis = dataflowcontrol.EnvireVisualization()
    nodes = {
        "linear": LinearDFN(),
        "square": SquareDFN()
    }
    periods = {
        "linear": 0.001,
        "square": 0.001
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x"),
        ("square.y", "result.y")
    )
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods, vis)
    dfc.setup()
    diagrams.save_graph_png(dfc, "network.png")
    for i in range(10000):
        dfc.process_sample(timestamp=i, stream_name="log.x", sample=i)
    dfc.node_statistics_.print_statistics()


if __name__ == "__main__":
    main()
