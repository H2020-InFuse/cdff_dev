from collections import defaultdict


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
        Temporal periods after which the processing steps of nodes are triggered

    visualization : Visualization, optional (default: None)
        Visualization

    verbose : int, optional (default: 0)
        Verbosity level

    Attributes
    ----------
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

    connection_map : list
        A list of pairs of output node and connected input node. Full names
        of the form 'node_name.port_name' are used here.
    """
    def __init__(self, nodes, connections, periods, visualization=None, verbose=0):
        self.nodes = nodes
        self.connections = connections
        self.periods = periods
        self.visualization = visualization
        self.verbose = verbose

        self.output_ports_ = None
        self.input_ports_ = None
        self.log_ports_ = None
        self.result_ports_ = None
        self.connection_map = None

    def setup(self):
        self._configure_nodes()
        self._cache_ports()
        self._configure_periods()
        self._configure_connections()

    def _configure_nodes(self):
        for node in self.nodes.values():
            node.configure()

    def _cache_ports(self):
        self.output_ports_ = dict(
            (node_name, dict()) for node_name in self.nodes.keys())
        self.input_ports_ = dict(
            (node_name, dict()) for node_name in self.nodes.keys())
        self.log_ports_ = defaultdict(list)
        self.result_ports_ = defaultdict(list)

        for output_port, input_port in self.connections:
            node_name, port_name, port = self._check_port(output_port, "Output")
            if port is None:
                self.log_ports_[node_name].append(port_name)
            else:
                self.output_ports_[node_name][port_name] = port

            node_name, port_name, port = self._check_port(input_port, "Input")
            if port is None:
                self.result_ports_[node_name].append(port_name)
            else:
                self.input_ports_[node_name][port_name] = port

    def _check_port(self, port, port_type):
        if port_type not in ["Input", "Output"]:
            raise ValueError("port_type must be either 'Input' or 'Output'")

        node_name, port_name = port.split(".")
        if node_name not in self.nodes:
            return node_name, port_name, None

        node = self.nodes[node_name]
        method_name = port_name + port_type

        if port_type == "Input" and not hasattr(node, method_name):
            raise AttributeError(
                "Could not find %s.%s(...) corresponding to port %s"
                % (type(node), method_name, port_name))

        return node_name, port_name, getattr(node, method_name)

    def _configure_periods(self):
        if set(self.nodes.keys()) != set(self.periods.keys()):
            raise ValueError(
                "Mismatch between nodes and periods. Nodes: %s, periods: %s"
                % (sorted(self.nodes.keys()), sorted(self.periods.keys())))

        self.last_processed = {node: -1 for node in self.nodes.keys()}

    def _configure_connections(self):
        self.connection_map = defaultdict(list)
        for output_port, input_port in self.connections:
            self.connection_map[output_port].append(input_port)

    def process_sample(self, timestamp, stream_name, sample):
        """TODO document me

        Note that timestamps must be greater than or equal 0.
        """
        self._run_all_nodes_before(timestamp)
        node, port = stream_name.split(".")
        self._push_input(node, port, sample)

    def _run_all_nodes_before(self, timestamp):
        changed = True
        while changed:
            current_node, timestamp_before_process = self._get_next_node(timestamp)

            if timestamp_before_process <= timestamp:
                changed = True
                self.last_processed[current_node] = timestamp_before_process
                node = self.nodes[current_node]

                try:
                    # TODO how do we pass the current time to the node?
                    node.set_time(timestamp_before_process)
                except:
                    if self.verbose >= 1:
                        print(current_node + ".set_time(time) not implemented")

                node.process()

                outputs = self._pull_output(
                    current_node, timestamp_before_process)

                for port_name, sample in outputs.items():
                    self._push_input(current_node, port_name, sample)

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
                              self.periods[node_name])
            if next_timestamp < timestamp_before_process:
                timestamp_before_process = next_timestamp
                current_node = node_name

        return current_node, timestamp_before_process

    def _pull_output(self, node_name, timestamp):
        outputs = dict()
        for port_name, getter in self.output_ports_[node_name].items():
            if self.verbose >= 1:
                print("[DataFlowControl] getting %s" % port_name)
            sample = getter()
            self.visualization.report_node_output(
                node_name, port_name, sample, timestamp)
            outputs[port_name] = sample
        return outputs

    def _push_input(self, node_name, port_name, sample):
        output_port = node_name + "." + port_name
        if output_port in self.connection_map:
            for input_port in self.connection_map[output_port]:
                if self.verbose >= 1:
                    print("[DataFlowControl] setting %s" % input_port)
                input_node_name, input_port_name = input_port.split(".")
                if input_node_name in self.input_ports_:
                    setter = self.input_ports_[input_node_name][input_port_name]
                    setter(sample)
                elif input_node_name in self.result_ports_:
                    pass  # TODO how should we handle result ports?
                else:
                    raise ValueError("Unknown node: '%s'" % input_node_name)
        elif self.verbose >= 2:
            print("[DataFlowControl] port '%s' is not connected"
                  % output_port)

    def ports(self):
        """TODO document me"""
        return (self.input_ports_, self.output_ports_, self.log_ports_,
                self.result_ports_)


from . import diagrams


class TextVisualization:
    """Text "visualization".

    This is just for debugging purposes.
    """
    def __init__(self):
        pass

    def report_dfc_network(self, dfc, network_visualization_filename):
        """TODO seems out of place here..."""
        diagrams.save_graph_png(dfc, network_visualization_filename)

    def report_node_output(self, node_name, port_name, sample, timestamp):
        print("Visualizing sample:")
        print("  Timestamp: %s" % timestamp)
        print("  Node: %s" % node_name)
        print("  Port: %s" % port_name)
        print("  Sample: %s" % sample)


# TODO envire visualizer
#self._update_envire_item(sample, port_name, timestamp)
