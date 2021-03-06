from cdff_dev import dataflowcontrol
from collections import defaultdict
from nose.tools import assert_in, assert_equal, assert_raises_regexp


class LinearDFN:
    def __init__(self):
        self.w = 2.0
        self.b = 1.0
        self.x = 0.0

    def set_configuration_file(self, filename):
        pass

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

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def xInput(self, x):
        self.x = x

    def process(self):
        self.y = self.x ** 2

    def yOutput(self):
        return self.y


class SquareDFPC:
    def __init__(self):
        self.x = 0.0

    def set_configuration_file(self, filename):
        pass

    def setup(self):
        pass

    def xInput(self, x):
        self.x = x

    def run(self):
        self.y = self.x ** 2

    def yOutput(self):
        return self.y


def test_wrong_stream_name_pattern():
    dfc = dataflowcontrol.DataFlowControl(
        nodes={"linear": LinearDFN()},
        connections=(("log/x", "linear.x"), ("linear.y", "result.y")),
        periods={}, trigger_ports={"linear": "x"}
    )
    assert_raises_regexp(
        ValueError, "Stream name must have the form", dfc.setup)


def test_stream_name_alias():
    dfc = dataflowcontrol.DataFlowControl(
        nodes={"linear": LinearDFN()},
        connections=(("log/x", "linear.x"), ("linear.y", "result.y")),
        periods={}, trigger_ports={"linear": "x"},
        stream_aliases={"log/x": "log.x"}
    )
    dfc.setup()


def test_missing_trigger():
    dfc = dataflowcontrol.DataFlowControl(
        nodes={"linear": LinearDFN()},
        connections=(("log.x", "linear.x"), ("linear.y", "result.y")),
        periods={}, trigger_ports={}
    )
    assert_raises_regexp(
        ValueError, "Mismatch between nodes and triggered nodes", dfc.setup)


def test_trigger_does_not_exist():
    dfc = dataflowcontrol.DataFlowControl(
        nodes={"linear": LinearDFN()},
        connections=(("log.x", "linear.x"), ("linear.y", "result.y")),
        periods={}, trigger_ports={"linear": "y"}
    )
    assert_raises_regexp(
        ValueError, "Trigger port .* does not exist", dfc.setup)


def test_triggered_twice():
    dfc = dataflowcontrol.DataFlowControl(
        nodes={"linear": LinearDFN()},
        connections=(("log.x", "linear.x"), ("linear.y", "result.y")),
        periods={"linear": 0.01}, trigger_ports={"linear": "x"}
    )
    assert_raises_regexp(ValueError, "", dfc.setup)


def test_smoke_setup():
    dfc = dataflowcontrol.DataFlowControl(
        nodes={"linear": LinearDFN()},
        connections=(("log.x", "linear.x"), ("linear.y", "result.y")),
        periods={"linear": 0.01}, trigger_ports={}
    )
    dfc.setup()
    dfc = dataflowcontrol.DataFlowControl(
        nodes={"linear": LinearDFN()},
        connections=(("log.x", "linear.x"), ("linear.y", "result.y")),
        periods={}, trigger_ports={"linear": "x"}
    )
    dfc.setup()


def test_dfc_periodic():
    nodes = {
        "linear": LinearDFN(),
        "square": SquareDFN()
    }
    periods = {
        "linear": 0.000001,
        "square": 0.000001
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x"),
        ("square.y", "result.y")
    )
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods=periods)
    dfc.setup()
    assert_equal(len(dfc.input_ports_), 2)
    assert_equal(len(dfc.output_ports_), 2)
    assert_equal(len(dfc.log_ports_), 1)
    assert_equal(len(dfc.result_ports_), 1)
    assert_equal(len(dfc.connection_map_), 3)

    vis = dataflowcontrol.NoVisualization()
    dfc.set_visualization(vis)
    for i in range(101):
        dfc.process_sample(timestamp=i, stream_name="log.x", sample=i)
    dfc.process(timestamp=102)
    assert_in("linear.y", vis.data)
    assert_equal(vis.data["linear.y"][0], 201.0)
    assert_in("square.y", vis.data)
    assert_equal(vis.data["square.y"][0], 40401.0)


def test_dfc_port_triggered():
    nodes = {
        "linear": LinearDFN(),
        "square": SquareDFN()
    }
    trigger_ports = {
        "linear": "x",
        "square": "x"
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x"),
        ("square.y", "result.y")
    )
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, trigger_ports=trigger_ports)
    dfc.setup()
    assert_equal(len(dfc.input_ports_), 2)
    assert_equal(len(dfc.output_ports_), 2)
    assert_equal(len(dfc.log_ports_), 1)
    assert_equal(len(dfc.result_ports_), 1)
    assert_equal(len(dfc.connection_map_), 3)

    vis = dataflowcontrol.NoVisualization()
    dfc.set_visualization(vis)
    for i in range(101):
        dfc.process_sample(timestamp=i, stream_name="log.x", sample=i)
    dfc.process(timestamp=102)
    assert_in("linear.y", vis.data)
    assert_equal(vis.data["linear.y"][0], 201.0)
    assert_in("square.y", vis.data)
    assert_equal(vis.data["square.y"][0], 40401.0)


def test_dfc_detects_node_that_is_not_dfn():
    class NoDFN:
        pass
    nodes = {
        "nodfn": NoDFN()
    }
    dfc = dataflowcontrol.DataFlowControl(nodes, (), periods={"nodfn": 1.0})
    assert_raises_regexp(ValueError, "is not a DFN", dfc.setup)


def test_dfc_periodic_realtime():
    nodes = {
        "linear": LinearDFN(),
        "square": SquareDFN()
    }
    periods = {
        "linear": 0.000001,
        "square": 0.000001
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x"),
        ("square.y", "result.y")
    )
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods=periods, real_time=True)
    dfc.setup()

    for i in range(101):
        dfc.process_sample(timestamp=i, stream_name="log.x", sample=i)
    dfc.process(timestamp=102)


def test_dfc_converts_dfpc():
    nodes = {
        "linear": LinearDFN(),
        "square": SquareDFPC()
    }
    trigger_ports = {
        "linear": "x",
        "square": "x"
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x"),
        ("square.y", "result.y")
    )
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, trigger_ports=trigger_ports)
    dfc.setup()
    assert_equal(len(dfc.input_ports_), 2)
    assert_equal(len(dfc.output_ports_), 2)
    assert_equal(len(dfc.log_ports_), 1)
    assert_equal(len(dfc.result_ports_), 1)
    assert_equal(len(dfc.connection_map_), 3)

    vis = dataflowcontrol.NoVisualization()
    dfc.set_visualization(vis)
    for i in range(101):
        dfc.process_sample(timestamp=i, stream_name="log.x", sample=i)
    dfc.process(timestamp=102)
    assert_in("linear.y", vis.data)
    assert_equal(vis.data["linear.y"][0], 201.0)
    assert_in("square.y", vis.data)
    assert_equal(vis.data["square.y"][0], 40401.0)


class MultiLinearDFN:
    def __init__(self):
        self.w = 2.0
        self.b = 1.0
        self.x = 0.0

    def set_configuration_file(self, filename):
        pass

    def configure(self):
        pass

    def xInput(self, x):
        self.x = x

    def process(self):
        self.y = self.w * self.x + self.b

    def yOutput(self):
        return self.y

    def zOutput(self):
        return self.y


def test_dfc_collects_all_output_ports():
    nodes = {
        "linear": MultiLinearDFN(),
        "square": SquareDFPC()
    }
    trigger_ports = {
        "linear": "x",
        "square": "x"
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x"),
        ("square.y", "result.y")
    )
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, trigger_ports=trigger_ports)
    dfc.setup()
    assert_equal(dfc.output_ports_["linear"], {"y", "z"})


class Logger(dataflowcontrol.LoggerBase):
    def __init__(self):
        self.results = defaultdict(lambda: [])

    def report_node_output(self, port_name, sample, timestamp):
        self.results[port_name].append((timestamp, sample))


def test_output_logging():
    nodes = {
        "linear": LinearDFN(),
        "square": SquareDFN()
    }
    periods = {
        "linear": 0.000001,
        "square": 0.000001
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x"),
        ("square.y", "result.y")
    )
    dfc = dataflowcontrol.DataFlowControl(nodes, connections, periods=periods)
    dfc.setup()
    assert_equal(len(dfc.input_ports_), 2)
    assert_equal(len(dfc.output_ports_), 2)
    assert_equal(len(dfc.log_ports_), 1)
    assert_equal(len(dfc.result_ports_), 1)
    assert_equal(len(dfc.connection_map_), 3)

    log = Logger()
    dfc.register_logger(log)
    for i in range(101):
        dfc.process_sample(timestamp=i, stream_name="log.x", sample=i)
    dfc.process(timestamp=102)
    assert_in("linear.y", log.results)
    assert_in("square.y", log.results)
    assert_equal(len(log.results), 2)
    assert_equal(log.results["linear.y"][0][0], 1)
    assert_equal(log.results["linear.y"][0][1], 1)
    assert_equal(log.results["square.y"][0][0], 1)
    assert_equal(log.results["square.y"][0][1], 1)
    assert_equal(log.results["linear.y"][1][0], 2)
    assert_equal(log.results["linear.y"][1][1], 3)
    assert_equal(log.results["square.y"][1][0], 2)
    assert_equal(log.results["square.y"][1][1], 9)


def test_memory_profiler():
    nodes = {
        "linear": LinearDFN(),
        "square": SquareDFN()
    }
    periods = {
        "linear": 0.000001,
        "square": 0.000001
    }
    connections = (
        ("log.x", "linear.x"),
        ("linear.y", "square.x"),
        ("square.y", "result.y")
    )
    dfc = dataflowcontrol.DataFlowControl(
        nodes, connections, periods=periods, memory_profiler=True)
    dfc.setup()

    for i in range(101):
        dfc.process_sample(timestamp=i, stream_name="log.x", sample=i)

    assert_equal(len(dfc.node_statistics_.configure_memory), 2)
    assert_equal(len(dfc.node_statistics_.process_memory), 2)
