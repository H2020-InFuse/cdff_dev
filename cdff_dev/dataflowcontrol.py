from collections import defaultdict
from abc import ABCMeta, abstractmethod
import time
import warnings


class DataFlowControl:
    """Data flow control for replaying logfiles.

    Parameters
    ----------
    nodes : dict
        Mapping from names to data fusion nodes

    connections : iterable of tuples
        Connections between ports of nodes. Each entry is a pair of input
        port identifier and output port identifier. Port identifiers are
        strings with component name and port name connected by a '.', for
        example, 'mynode.motion_command'. Stream names from logfiles can
        also be port identifiers.

    periods : dict
        Temporal interval after which the processing steps of nodes are
        triggered. The time interval must be given in seconds. Each node
        must be either periodically triggered or triggered when input on
        a defined port arrives.

    trigger_ports : dict
        For each node the port that trigger their execution.

    real_time : bool, optional (default: False)
        Run logs in real time. This is a very simple implementation that
        tries to sleep for the right amount of time before log data is
        replayed. If any DFN takes longer than allowed, the whole processing
        network will be delayed and slower than real time.

    verbose : int, optional (default: 0)
        Verbosity level

    Attributes
    ----------
    node_statistics_ : NodeStatistics
        Store runtime data about nodes that can be used for analysis of the
        network.

    output_ports_ : dict, optional (default: {})
        A dictionary with node names as keys and dictionaries as values. The
        values represent a dictionary that maps from port names (without
        node prefix) to the corresponding function of the node implementation.

    input_ports_ : dict, optional (default: {})
        A dictionary with node names as keys and dictionaries as values. The
        values represent a dictionary that maps from port names (without
        node prefix) to the corresponding function of the node implementation.

    log_ports_ : dict
        A dictionary with node names as keys and port names (without
        node prefix) as values.

    result_ports_ : dict
        A dictionary with node names as keys and port names (without
        node prefix) as values.

    connection_map_ : list
        A list of pairs of output node and connected input node. Full names
        of the form 'node_name.port_name' are used here.
    """
    def __init__(self, nodes, connections, periods=None, trigger_ports=None,
                 real_time=False, verbose=0):
        self.nodes = nodes
        self.connections = connections
        self.periods = periods
        self.trigger_ports = trigger_ports
        self.real_time = real_time
        self.verbose = verbose

        self.visualization = None
        self.node_statistics_ = None
        self.output_ports_ = None
        self.input_ports_ = None
        self.log_ports_ = None
        self.result_ports_ = None
        self.connection_map_ = None

        self._node_facade = None

        self._last_timestamp = None
        self._real_start_time = None

    def setup(self):
        """Setup network.

        This function must be called before any log data can be fed into this
        class. Initializes internal data structures and configures nodes.
        """
        self.node_statistics_ = NodeStatistics()
        self._node_facade = NodeFacade(self.nodes, self.verbose)
        self._node_facade.configure_all()
        self._cache_ports()
        self._configure_triggers()
        self._configure_connections()
        self._last_timestamp = None

    def set_visualization(self, visualization):
        """Set visualization.

        Parameters
        ----------
        visualization : subclass of VisualizationBase
            Visualization of replayed log samples or results of processing
            steps
        """
        self.visualization = visualization

    def register_world_state(self, world_state):
        """Register a world state representation.

        The EnviRe graph from the world state representation will be
        used by the transformer node if there is one. Hence, the
        transformations in the world state reflect the current state
        of the world.
        """
        self._node_facade.register_graph(world_state.graph_)

    def _cache_ports(self):
        """Cache setters and getters for ports."""
        self.output_ports_ = defaultdict(list)
        self.input_ports_ = defaultdict(list)
        self.log_ports_ = defaultdict(list)
        self.result_ports_ = defaultdict(list)

        for output_port, input_port in self.connections:
            node_name, port_name, port_exists = self._node_facade.check_port(
                output_port, "Output")
            if port_exists:
                self.output_ports_[node_name].append(port_name)
            else:
                self.log_ports_[node_name].append(port_name)

            node_name, port_name, port_exists = self._node_facade.check_port(
                input_port, "Input")
            if port_exists:
                self.input_ports_[node_name].append(port_name)
            else:
                self.result_ports_[node_name].append(port_name)

    def _configure_triggers(self):
        """Check and save periods for nodes."""
        if self.periods is None:
            self.periods = {}
        if self.trigger_ports is None:
            self.trigger_ports = {}

        self.__check_all_nodes_are_triggered()
        self.__check_trigger_ports_exist()

        self.periods_microseconds = {
            node_name: max(int(1e6 * period), 1)
            for node_name, period in self.periods.items()}
        self.last_processed = {node: -1 for node in self.periods.keys()}

    def __check_all_nodes_are_triggered(self):
        all_nodes = self._node_facade.node_names()
        triggered_nodes = set(self.periods.keys()).union(
            set(self.trigger_ports.keys()))
        if set(all_nodes) != set(triggered_nodes):
            raise ValueError(
                "Mismatch between nodes and triggered nodes. "
                "Nodes: %s, triggered nodes: %s"
                % (sorted(all_nodes), sorted(triggered_nodes)))

    def __check_trigger_ports_exist(self):
        for node_name, port_name in self.trigger_ports.items():
            input_ports = self.input_ports_[node_name]
            if port_name not in input_ports:
                raise ValueError(
                    "Trigger port '%s.%s' does not exist. Node only has the "
                    "following ports: %s"
                    % (node_name, port_name, ", ".join(input_ports)))

    def _configure_connections(self):
        """Initialize connections."""
        self.connection_map_ = defaultdict(list)
        for output_port, input_port in self.connections:
            self.connection_map_[output_port].append(input_port)

    def process_sample(self, timestamp, stream_name, sample):
        """Makes a new sample available to the network and runs nodes.

        Parameters
        ----------
        timestamp : int
            Current time from the log data. Note that timestamps must be
            greater than or equal 0 and must be given in microseconds.

        stream_name : str
            Name of the stream in the form 'node.port'.

        sample : CDFF type
            Current sample. The type must correspond to the connected input
            port. Only CDFF types are allowed.
        """
        self._run_all_nodes_before(timestamp)

        if self.real_time and self._last_timestamp is not None:
            self._real_start_time = time.time()

        if self.visualization is not None:
            self.visualization.report_node_output(
                stream_name, sample, timestamp)

        if self.real_time and self._last_timestamp is not None:
            if self._last_timestamp is not None:
                processing_time = time.time() - self._real_start_time
                time_between_samples = float(
                    timestamp - self._last_timestamp) / 1e6
                sleep_time = time_between_samples - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    warnings.warn(
                        "Processing took too long, %.3f behind real time "
                        "schedule" % -sleep_time)

        self._push_input(stream_name, sample, timestamp)

        if self.real_time:
            self._last_timestamp = timestamp

    def process(self, timestamp):
        """Runs all nodes until a specific timestamp.

        Parameters
        ----------
        timestamp : int
            Current time from the log data. Note that timestamps must be
            greater than or equal 0.
        """
        self._run_all_nodes_before(timestamp)

    def _run_all_nodes_before(self, timestamp):
        """Run all nodes that should have been executed before given time.

        Parameters
        ----------
        timestamp : int
            Simulation time
        """
        changed = True
        while changed:
            current_node, timestamp_before_process = self._get_next_node(
                timestamp)

            if timestamp_before_process <= timestamp:
                changed = True
                self.last_processed[current_node] = timestamp_before_process
                self._process_node(current_node, timestamp_before_process)
                self._post_processing(current_node, timestamp_before_process)
            else:
                changed = False

    def _get_next_node(self, timestamp):
        """Get node that will be executed next."""
        timestamp_before_process = float("inf")
        current_node = None
        # Nodes will be sorted alphabetically to ensure reproducibility
        all_nodes = sorted(self.last_processed.keys())
        for node_name in all_nodes:
            if self.last_processed[node_name] < 0:
                self.last_processed[node_name] = timestamp

            next_timestamp = (self.last_processed[node_name] +
                              self.periods_microseconds[node_name])
            if next_timestamp < timestamp_before_process:
                timestamp_before_process = next_timestamp
                current_node = node_name

        return current_node, timestamp_before_process

    def _post_processing(self, node, timestamp):
        outputs = self._pull_output(node)

        if self.visualization is not None:
            for port_name, sample in outputs.items():
                # TODO should we add the processing time?
                self.visualization.report_node_output(
                    port_name, sample, timestamp)

        for port_name, sample in outputs.items():
            self._push_input(port_name, sample, timestamp)

    def _pull_output(self, node_name):
        """Get all outputs from a given node."""
        outputs = dict()
        for port_name in self.output_ports_[node_name]:
            if self.verbose >= 1:
                print("[DataFlowControl] getting %s" % port_name)
            sample = self._node_facade.read_output_port(node_name, port_name)
            outputs[node_name + "." + port_name] = sample
        return outputs

    def _push_input(self, output_port, sample, timestamp):
        """Push input to given port."""
        if output_port in self.connection_map_:
            for input_port in self.connection_map_[output_port]:
                if self.verbose >= 1:
                    print("[DataFlowControl] setting %s" % input_port)
                input_node_name, input_port_name = input_port.split(".")
                if input_node_name in self.input_ports_:
                    self._node_facade.write_input_port(
                        input_node_name, input_port_name, sample)
                    self._port_trigger(
                        input_node_name, input_port_name, timestamp)
                elif input_node_name in self.result_ports_:
                    pass  # TODO how should we handle result ports?
                else:
                    raise ValueError("Unknown node: '%s'" % input_node_name)
        elif self.verbose >= 2:
            print("[DataFlowControl] port '%s' is not connected"
                  % output_port)

    def _port_trigger(self, node_name, port_name, timestamp):
        if (node_name in self.trigger_ports and
                port_name in self.trigger_ports[node_name]):
            self._process_node(node_name, timestamp)
            self._post_processing(node_name, timestamp)

    def _process_node(self, node_name, timestamp):
        self._node_facade.set_time(node_name, timestamp)

        processing_time = self._node_facade.process(node_name)

        self.node_statistics_.report_processing_duration(
            node_name, processing_time)
        if self.verbose >= 1:
            print("[DataFlowControl] Processed node '%s' in %g seconds"
                    % (node_name, processing_time))


class NodeFacade:
    """Decouples the node interface from data flow control.

    Parameters
    ----------
    nodes : dict
        Mapping from node names to node instances

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(self, nodes, verbose=0):
        self.nodes = nodes
        self.verbose = verbose

    def configure_all(self):
        for node in self.nodes.values():
            node.configure()

    def register_graph(self, graph):
        """Assign EnviRe graph to a node with the name 'transformer'.

        Will do nothing if there is no transformer. We assume that the
        transformer has the field 'graph_'. See architecture documentation
        about EnviRe integration for more details.
        """
        if self.exists("transformer"):
            self.nodes["transformer"].graph_ = graph

    def node_names(self):
        return self.nodes.keys()

    def exists(self, node_name):
        return node_name in self.nodes

    def process(self, node_name):
        start_time = time.process_time()

        self.nodes[node_name].process()

        end_time = time.process_time()
        processing_time = end_time - start_time
        return processing_time

    def set_time(self, node_name, timestamp):
        node = self.nodes[node_name]
        try:
            # TODO how do we pass the current time to the node?
            node.set_time(timestamp)
        except AttributeError:
            if self.verbose >= 1:
                print(node_name + ".set_time(time) not implemented")

    def check_port(self, port, port_type):
        if port_type not in ["Input", "Output"]:
            raise ValueError("port_type must be either 'Input' or 'Output'")

        node_name, port_name = port.split(".")
        if not self.exists(node_name):
            return node_name, port_name, False

        node = self.nodes[node_name]
        method_name = port_name + port_type

        if port_type == "Input" and not hasattr(node, method_name):
            raise AttributeError(
                "Could not find %s.%s(...) corresponding to port %s"
                % (node.__class__.__name__, method_name, port_name))

        return node_name, port_name, True

    def read_output_port(self, node_name, port_name):
        node = self.nodes[node_name]
        method_name = port_name + "Output"
        getter = getattr(node, method_name)
        return getter()

    def write_input_port(self, node_name, port_name, sample):
        node = self.nodes[node_name]
        method_name = port_name + "Input"
        setter = getattr(node, method_name)
        return setter(sample)


class NodeStatistics:
    """Collects statistics about nodes.

    Attributes
    ----------
    processing_durations_ : dict
        Contains processing times for each node that has been executed.
        Only the time in this process is measured.
    """
    def __init__(self):
        self.processing_durations_ = defaultdict(list)

    def report_processing_duration(self, node_name, duration):
        self.processing_durations_[node_name].append(duration)

    def print_statistics(self):
        average_processing_durations = dict(
            (node_name, sum(self.processing_durations_[node_name]) /
             len(self.processing_durations_[node_name]))
            for node_name in self.processing_durations_.keys()
        )

        for node_name, average_processing_duration in \
                self.processing_durations_.items():
            print("Node: %s" % node_name)
            print("Average processing time: %g s"
                  % average_processing_durations[node_name])
            print("Number of calls: %d"
                  % len(self.processing_durations_[node_name]))


class VisualizationBase(metaclass=ABCMeta):
    @abstractmethod
    def report_node_output(self, port_name, sample, timestamp):
        """Report log sample or result of processing step to visualization.

        Parameters
        ----------
        port_name : str
            Name of an output port or log stream

        sample : object
            Log sample or result of processing step

        timestamp : int
            Current replay time
        """


class NoVisualization(VisualizationBase):
    """No visualization.

    This is just for testing purposes.
    """
    def __init__(self):
        self.data = dict()

    def report_node_output(self, port_name, sample, timestamp):
        self.data[port_name] = (sample, timestamp)


class TextVisualization(VisualizationBase):
    """Text "visualization".

    This is just for debugging purposes.
    """
    def __init__(self):
        pass

    def report_node_output(self, port_name, sample, timestamp):
        print("Visualizing sample:")
        print("  Timestamp: %s" % timestamp)
        print("  Port: %s" % port_name)
        print("  Sample: %s" % sample)
