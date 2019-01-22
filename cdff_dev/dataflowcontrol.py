from . import dfnhelpers
from collections import defaultdict
from abc import ABCMeta, abstractmethod
import inspect
import time
import warnings
import memory_profiler


class DataFlowControl:
    """Data flow control for replaying logfiles.

    We will run DFNs and DFCPs in the correct chronological order while
    replaying log files. You can register visualization instances and / or
    logger instances to display or output results of DFNs or DFPCs.
    It is also possible to visualize the connections between log streams,
    DFN ports, and DFPC ports as a graph.

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
        For each node the port that triggers its execution.

    real_time : bool, optional (default: False)
        Run logs in real time. This is a very simple implementation that
        tries to sleep for the right amount of time before log data is
        replayed. If any DFN takes longer than allowed, the whole processing
        network will be delayed and slower than real time.

    stream_aliases : dict, optional (default: {})
        Mapping from original stream names to their aliases if they have any.

    memory_profiler : bool, optional (default: False)
        Activate memory profiler for DFNs.

    verbose : int, optional (default: 0)
        Verbosity level

    Attributes
    ----------
    visualizations_ : list
        Registered visualizations.

    loggers_ : list
        Registered loggers.

    node_statistics_ : NodeStatistics
        Store runtime data about nodes that can be used for analysis of the
        network.

    output_ports_ : dict, optional (default: {})
        A dictionary with node names as keys and a set of output port names
        (without node prefix) as values.

    input_ports_ : dict, optional (default: {})
        A dictionary with node names as keys and a set of input port names
        (without node prefix) as values.

    log_ports_ : dict
        A dictionary with node names as keys and a set of log port names
        (without node prefix) as values.

    result_ports_ : dict
        A dictionary with node names as keys and a set of result port names
        (without node prefix) as values.

    connection_map_ : list
        A list of pairs of output node and connected input node. Full names
        of the form 'node_name.port_name' are used here.
    """
    def __init__(self, nodes, connections, periods=None, trigger_ports=None,
                 real_time=False, stream_aliases=None, memory_profiler=False,
                 verbose=0):
        self.nodes = nodes
        self.connections = connections
        self.periods = periods
        self.trigger_ports = trigger_ports
        self.real_time = real_time
        self.stream_aliases = stream_aliases
        self.memory_profiler = memory_profiler
        self.verbose = verbose

        self.visualizations_ = []
        self.loggers_ = []
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
        if self.stream_aliases is None:
            self.stream_aliases = {}

        self.node_statistics_ = NodeStatistics()
        self._node_facade = NodeFacade(
            self.nodes, self.memory_profiler, self.verbose)
        results = self._node_facade.configure_all()
        if results is not None:
            self.node_statistics_.report_configure_memory(results)
        self._cache_ports()
        self._configure_triggers()
        self._configure_connections()
        self._last_timestamp = None

    def register_visualization(self, visualization):
        """Register visualization.

        Parameters
        ----------
        visualization : subclass of VisualizationBase
            Visualization of replayed log samples or results of processing
            steps
        """
        self.visualizations_.append(visualization)

    def register_logger(self, logger):
        """Register visualization.

        Parameters
        ----------
        logger : subclass of LoggerBase
            Logger for results of processing steps
        """
        self.loggers_.append(logger)

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
        self.output_ports_ = defaultdict(set)
        self.input_ports_ = defaultdict(set)
        self.log_ports_ = defaultdict(set)
        self.result_ports_ = defaultdict(set)

        for node_name in self.nodes.keys():
            for port_name in self._node_facade.output_ports(node_name):
                self.output_ports_[node_name].add(port_name)

        for output_port, input_port in self.connections:
            input_port = self._check_stream_name(input_port)
            output_port = self._check_stream_name(output_port)

            node_name, port_name, port_exists = self._node_facade.check_port(
                output_port, "Output")
            if port_exists:
                assert port_name in self.output_ports_[node_name]
            else:
                self.log_ports_[node_name].add(port_name)

            node_name, port_name, port_exists = self._node_facade.check_port(
                input_port, "Input")
            if port_exists:
                self.input_ports_[node_name].add(port_name)
            else:
                self.result_ports_[node_name].add(port_name)

    def _configure_triggers(self):
        """Check and save periods for nodes."""
        if self.periods is None:
            self.periods = {}
        if self.trigger_ports is None:
            self.trigger_ports = {}

        self.__check_all_nodes_are_triggered_once()
        self.__check_trigger_ports_exist()

        self.periods_microseconds = {
            node_name: max(int(1e6 * period), 1)
            for node_name, period in self.periods.items()}
        self.last_processed = {node: -1 for node in self.periods.keys()}

    def __check_all_nodes_are_triggered_once(self):
        all_nodes = self._node_facade.node_names()
        duplicate_triggered_nodes = set(
            self.periods.keys()).intersection(set(self.trigger_ports.keys()))
        if duplicate_triggered_nodes:
            raise ValueError(
                "Nodes must not be port-triggered and periodically triggered "
                "at the same time, but the following nodes are: %s"
                % (duplicate_triggered_nodes,))
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
        stream_name = self._check_stream_name(stream_name)
        self.process(timestamp)
        self._sleep_realtime(timestamp)
        self._push_input(stream_name, sample, timestamp)

    def _check_stream_name(self, stream_name):
        """Check if stream name satisfies the pattern '<node>.<port>'.

        This function also supports stream name aliases.
        """
        if stream_name in self.stream_aliases:
            stream_name = self.stream_aliases[stream_name]

        if "." not in stream_name:
            raise ValueError(
                "Stream name must have the form '<node>.<port>', got '%s'."
                % stream_name)

        return stream_name

    def _sleep_realtime(self, timestamp):
        """Sleep to ensure real time replay."""
        if self.real_time and self._last_timestamp is not None:
            self._real_start_time = time.time()
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
            self._last_timestamp = timestamp

    def process(self, timestamp):
        """Run all nodes that should have been executed before given time.

        Parameters
        ----------
        timestamp : int
            Current time from the log data. Note that timestamps must be
            greater than or equal 0.
        """
        changed = True
        while changed:
            current_node, timestamp_before_process = self._get_next_node(
                timestamp)

            if timestamp_before_process <= timestamp:
                changed = True
                self.last_processed[current_node] = timestamp_before_process
                self._process_node(current_node, timestamp_before_process)
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

    def _process_node(self, node_name, timestamp):
        """Execute node and pass outputs to next nodes."""
        self._node_facade.set_time(node_name, timestamp)

        results = self._node_facade.process(node_name)

        if self.memory_profiler:
            processing_time, memory = results
            self.node_statistics_.report_process_memory(node_name, memory)
        else:
            processing_time = results
        self.node_statistics_.report_processing_duration(
            node_name, processing_time)
        if self.verbose >= 1:
            print("[DataFlowControl] Processed node '%s' in %g seconds"
                  % (node_name, processing_time))

        outputs = self._pull_output(node_name)
        for port_name, sample in outputs.items():
            for logger in self.loggers_:
                logger.report_node_output(port_name, sample, timestamp)
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
        for visualization in self.visualizations_:
            visualization.report_node_output(output_port, sample, timestamp)

        if output_port in self.connection_map_:
            for input_port in self.connection_map_[output_port]:
                if self.verbose >= 1:
                    print("[DataFlowControl] setting %s" % input_port)
                input_node_name, input_port_name = input_port.split(".")
                if input_node_name in self.input_ports_:
                    try:
                        self._node_facade.write_input_port(
                            input_node_name, input_port_name, sample)
                    except TypeError as e:
                        warnings.warn("Failed to write on input port %s.%s. "
                                      "Forwarding error message."
                                      % (input_node_name, input_port_name))
                        raise e
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
        """Run port-triggered nodes recursively."""
        if (node_name in self.trigger_ports and
                port_name in self.trigger_ports[node_name]):
            self._process_node(node_name, timestamp)

    # for backward compatibility
    set_visualization = register_visualization


class NodeFacade:
    """Decouples the node interface from data flow control.

    Parameters
    ----------
    nodes : dict
        Mapping from node names to node instances

    memory_profiler : bool, optional (default: False)
        Activate memory profiler for DFNs.

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(self, nodes, memory_profiler, verbose=0):
        self.nodes = nodes
        self.memory_profiler = memory_profiler
        self.verbose = verbose

    def configure_all(self):
        if self.memory_profiler:
            self.memory_profiler_ = MemoryProfiler(include_children=True)
            self.configure_results = dict()

        for name in sorted(self.nodes.keys()):
            if not dfnhelpers.isdfn(self.nodes[name], verbose=self.verbose):
                if dfnhelpers.isdfpc(self.nodes[name], verbose=self.verbose):
                    self.nodes[name] = dfnhelpers.wrap_dfpc_as_dfn(
                        self.nodes[name])
                else:
                    raise ValueError("'%s' is not a DFN." % name)

            if self.memory_profiler:
                self.memory_profiler_.prepare_memory_profiling(
                    "%s configure" % name)

            try:
                self.nodes[name].configure()
            except:
                warnings.warn("Configuring %s failed" % name)

            if self.memory_profiler:
                mem = self.memory_profiler_.memory_profile(
                    "%s configure" % name)
                self.configure_results[name] = mem

        if self.memory_profiler:
            return self.configure_results

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

        if self.memory_profiler:
            self.memory_profiler_.prepare_memory_profiling(
                "%s process" % node_name)

        self.nodes[node_name].process()

        if self.memory_profiler:
            mem = self.memory_profiler_.memory_profile("%s process" % node_name)

        end_time = time.process_time()
        processing_time = end_time - start_time
        if self.memory_profiler:
            return processing_time, mem
        else:
            return processing_time

    def set_time(self, node_name, timestamp):
        node = self.nodes[node_name]
        try:
            # TODO how do we pass the current time to the node?
            node.set_time(timestamp)
        except AttributeError:
            if self.verbose >= 1:
                print(node_name + ".set_time(time) not implemented")

    def output_ports(self, node_name):
        result = inspect.getmembers(self.nodes[node_name],
                                    predicate=dfnhelpers.isoutput)
        return [port_name.replace("Output", "") for port_name, _ in result]

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
        self.process_memory = defaultdict(list)
        self.configure_memory = dict()

    def report_process_memory(self, node_name, memory):
        self.process_memory[node_name].append(memory)

    def report_configure_memory(self, memories):
        self.configure_memory.update(memories)

    def report_processing_duration(self, node_name, duration):
        self.processing_durations_[node_name].append(duration)

    def print_statistics(self):
        average_processing_durations = dict(
            (node_name, sum(self.processing_durations_[node_name]) /
             len(self.processing_durations_[node_name]))
            for node_name in self.processing_durations_.keys()
        )
        if self.process_memory:
            max_memory_added = dict(
                (node_name, max(self.process_memory[node_name]))
                for node_name in self.process_memory.keys()
            )
            average_memory_added = dict(
                (node_name, sum(self.process_memory.get(node_name, [])) /
                 len(self.process_memory.get(node_name, 1)))
                for node_name in self.process_memory.keys()
            )
            accumulated_memory_added = dict(
                (node_name, sum(self.process_memory[node_name]))
                for node_name in self.process_memory.keys()
            )

        print("=" * 80)
        print("    Node Statistics")
        print("=" * 80)

        for node_name, average_processing_duration in \
                self.processing_durations_.items():
            print("Node: %s" % node_name)
            print("  Average processing time: %g s"
                  % average_processing_durations[node_name])
            print("  Number of calls: %d"
                  % len(self.processing_durations_[node_name]))

            if not self.process_memory:
                continue

            print("  Memory:")
            print("  - added with configure: %g MiB"
                  % self.configure_memory[node_name])
            print("  - added with process (average): %g MiB"
                  % average_memory_added[node_name])
            print("  - added with process (maximum): %g MiB"
                  % max_memory_added[node_name])
            print("  - added with process (accumulated): %g MiB"
                  % accumulated_memory_added[node_name])


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


class LoggerBase(metaclass=ABCMeta):
    @abstractmethod
    def report_node_output(self, port_name, sample, timestamp):
        """Report result of processing step to logger.

        Parameters
        ----------
        port_name : str
            Name of an output port or log stream

        sample : object
            Log sample or result of processing step

        timestamp : int
            Current replay time
        """


class MemoryProfiler:
    """Simple memory profiler based on memory_profiler.

    For more details, see https://pypi.org/project/memory_profiler/
    """
    def __init__(self, backend="psutil", include_children=False):
        self.include_children = include_children
        self.backend = memory_profiler.choose_backend(backend)
        if (self.backend == 'tracemalloc' and
                memory_profiler.has_tracemalloc):
            if not memory_profiler.tracemalloc.is_tracing():
                memory_profiler.tracemalloc.start()
        self.mem_usage_before = dict()

    def __del__(self):
        if (self.backend == 'tracemalloc' and
                memory_profiler.has_tracemalloc and
                memory_profiler.tracemalloc.is_tracing()):
            memory_profiler.tracemalloc.stop()

    def prepare_memory_profiling(self, category):
        """Must be called before the function that we want to measure."""
        if category not in self.mem_usage_before:
            self.mem_usage_before[category] = 0.0
        self.mem_usage_before[category] = memory_profiler._get_memory(
            -1, self.backend, timestamps=False,
            include_children=self.include_children)

    def memory_profile(self, category):
        """Get the memory consumption of the function."""
        mem_usage_after = memory_profiler._get_memory(
            -1, self.backend, timestamps=False,
            include_children=self.include_children)
        return mem_usage_after - self.mem_usage_before[category]
