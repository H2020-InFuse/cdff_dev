from cdff_dev import dataflowcontrol


class LinearDFN:
    def __init__(self):
        self.w = 2.0
        self.b = 1.0

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
        pass

    def configure(self):
        pass

    def xInput(self, x):
        self.x = x

    def process(self):
        self.y = self.x ** 2

    def yOutput(self):
        return self.y


def main():
    vis = dataflowcontrol.TextVisualization()
    nodes = {
        "linear": LinearDFN(),
        "square": SquareDFN()
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x")
    )
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, vis)
    dfc.setup()
    for k, v in dfc.port_cache.items():
        print("%s: %s" % (k, v))


if __name__ == "__main__":
    main()
