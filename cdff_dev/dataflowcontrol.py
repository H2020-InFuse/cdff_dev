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
        triggered. The time interval must be given in seconds.

    real_time : bool, optional (default: False)
        Run logs in real time. This is a very simple implementation that
        tries to sleep for the right amount of time before log data is
        replayed. If any DFN takes longer than allowed this will be slower
        than real time.

    verbose : int, optional (default: 0)
        Verbosity level

    Attributes
    ----------
    node_statistics_ : NodeStatistics
        Store runtime data about nodes that can be used for analysis of the
        network.

    output_ports_ : dict
        A dictionary with node names as keys and dictionaries as values. The
        values represent a dictionary that maps from port names (without
        node prefix) to the corresponding function of the node implementation.

    input_ports_ : dict
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
    def __init__(self, nodes, connections, periods, real_time=False, verbose=0):
        self.nodes = nodes
        self.connections = connections
        self.periods = periods
        self.real_time = real_time
        self.verbose = verbose

        self.visualization = None
        self.node_statistics_ = None
        self.output_ports_ = None
        self.input_ports_ = None
        self.log_ports_ = None
        self.result_ports_ = None
        self.connection_map_ = None

        self.node_facade = None

        self.last_timestamp = None

    def setup(self):
        """Setup network.

        This function must be called before any log data can be fed into this
        class. Initializes internal data structures and configures nodes.
        """
        self.node_statistics_ = NodeStatistics()
        self.node_facade = NodeFacade(self.nodes, self.verbose)
        self.node_facade.configure_all()
        self._cache_ports()
        self._configure_periods()
        self._configure_connections()
        self.last_timestamp = None

    def set_visualization(self, visualization):
        """Set visualization.

        Parameters
        ----------
        visualization : subclass of VisualizationBase
            Visualization of replayed log samples or results of processing
            steps
        """
        self.visualization = visualization

    def _cache_ports(self):
        self.output_ports_ = defaultdict(list)
        self.input_ports_ = defaultdict(list)
        self.log_ports_ = defaultdict(list)
        self.result_ports_ = defaultdict(list)

        for output_port, input_port in self.connections:
            node_name, port_name, port_exists = self.node_facade.check_port(
                output_port, "Output")
            if port_exists:
                self.output_ports_[node_name].append(port_name)
            else:
                self.log_ports_[node_name].append(port_name)

            node_name, port_name, port_exists = self.node_facade.check_port(
                input_port, "Input")
            if port_exists:
                self.input_ports_[node_name].append(port_name)
            else:
                self.result_ports_[node_name].append(port_name)

    def _configure_periods(self):
        if set(self.node_facade.node_names()) != set(self.periods.keys()):
            raise ValueError(
                "Mismatch between nodes and periods. Nodes: %s, periods: %s"
                % (sorted(self.node_facade.node_names()),
                   sorted(self.periods.keys())))

        self.periods_microseconds = {
            node_name: max(int(1e6 * period), 1)
            for node_name, period in self.periods.items()}
        self.last_processed = {
            node: -1 for node in self.node_facade.node_names()}

    def _configure_connections(self):
        self.connection_map_ = defaultdict(list)
        for output_port, input_port in self.connections:
            self.connection_map_[output_port].append(input_port)

    def process_sample(self, timestamp, stream_name, sample):
        """Makes a new sample available to the network and runs nodes.

        Parameters
        ----------
        timestamp : int
            Current time from the log data. Note that timestamps must be
            greater than or equal 0.

        stream_name : str
            Name of the stream in the form 'node.port'.

        sample : anything
            Current sample. The type must correspond to the connected input
            port.
        """
        self._run_all_nodes_before(timestamp)

        if self.real_time and self.last_timestamp is not None:
            self.real_start_time = time.time()

        if self.visualization is not None:
            self.visualization.report_node_output(
                stream_name, sample, timestamp)

        if self.real_time and self.last_timestamp is not None:
            if self.last_timestamp is not None:
                processing_time = time.time() - self.real_start_time
                time_between_samples = float(
                    timestamp - self.last_timestamp) / 1000000.0
                sleep_time = time_between_samples - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    warnings.warn(
                        "Processing took too long, %.3f behind real time "
                        "schedule" % -sleep_time)

        self._push_input(stream_name, sample)

        if self.real_time:
            self.last_timestamp = timestamp

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
        changed = True
        while changed:
            current_node, timestamp_before_process = self._get_next_node(
                timestamp)

            if timestamp_before_process <= timestamp:
                changed = True
                self.last_processed[current_node] = timestamp_before_process

                self.node_facade.set_time(
                    current_node, timestamp_before_process)

                start_time = time.process_time()

                self.node_facade.process(current_node)

                end_time = time.process_time()
                processing_time = end_time - start_time
                self.node_statistics_.report_processing_duration(
                    current_node, processing_time)
                if self.verbose >= 1:
                    print("[DataFlowControl] Processed node '%s' in %g seconds"
                          % (current_node, processing_time))

                outputs = self._pull_output(current_node)

                if self.visualization is not None:
                    for port_name, sample in outputs.items():
                        # TODO should we add the processing time?
                        self.visualization.report_node_output(
                            port_name, sample, timestamp_before_process)

                for port_name, sample in outputs.items():
                    self._push_input(port_name, sample)

                # TODO use output_ports_ to implement port trigger
                # output_ports = self.output_ports_[current_node].keys()
            else:
                changed = False

    def _get_next_node(self, timestamp):
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

    def _pull_output(self, node_name):
        outputs = dict()
        for port_name in self.output_ports_[node_name]:
            if self.verbose >= 1:
                print("[DataFlowControl] getting %s" % port_name)
            sample = self.node_facade.read_output_port(node_name, port_name)
            outputs[node_name + "." + port_name] = sample
        return outputs

    def _push_input(self, output_port, sample):
        if output_port in self.connection_map_:
            for input_port in self.connection_map_[output_port]:
                if self.verbose >= 1:
                    print("[DataFlowControl] setting %s" % input_port)
                input_node_name, input_port_name = input_port.split(".")
                if input_node_name in self.input_ports_:
                    self.node_facade.write_input_port(
                        input_node_name, input_port_name, sample)
                elif input_node_name in self.result_ports_:
                    pass  # TODO how should we handle result ports?
                else:
                    raise ValueError("Unknown node: '%s'" % input_node_name)
        elif self.verbose >= 2:
            print("[DataFlowControl] port '%s' is not connected"
                  % output_port)


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

    def node_names(self):
        return self.nodes.keys()

    def exists(self, node_name):
        return node_name in self.nodes

    def process(self, node_name):
        self.nodes[node_name].process()

    def set_time(self, node_name, time):
        node = self.nodes[node_name]
        try:
            # TODO how do we pass the current time to the node?
            node.set_time(time)
        except:
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
                % (type(node), method_name, port_name))

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

        print("Processing times:")
        for node_name, average_processing_duration in \
                self.processing_durations_.items():
            print("  " + node_name + ": %g s"
                  % average_processing_durations[node_name])


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
