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
    """
    def __init__(self, nodes, connections, periods, visualization=None, verbose=0):
        self.nodes = nodes
        self.connections = connections
        self.periods = periods
        self.visualization = visualization
        self.verbose = verbose

    def setup(self):
        self._configure_nodes()
        self._cache_ports()
        self._configure_periods()

    def _configure_nodes(self):
        for node in self.nodes.values():
            node.configure()

    def _cache_ports(self):
        self.output_ports = dict(
            (node_name, dict()) for node_name in self.nodes.keys())
        self.input_ports = dict()
        for output_port, input_port in self.connections:
            node_name, port_name, port = self._check_port(output_port, "Output")
            if node_name is not None:
                self.output_ports[node_name][port_name] = port
            _, _, port = self._check_port(input_port, "Input")
            self.input_ports[input_port] = port

    def _check_port(self, port, port_type):
        if port_type not in ["Input", "Output"]:
            raise ValueError("port_type must be either 'Input' or 'Output'")

        node_name, port_name = port.split(".")
        if node_name not in self.nodes:
            if port_type == "Input":
                raise ValueError("Unknown node '%s' from port name '%s'."
                                 % (node_name, port))
            else:
                return None, None, None

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

    def process_sample(self, timestamp, stream_name, sample):
        """TODO document me

        Note that timestamps must be greater than or equal 0.
        """
        self._run_all_nodes_before(timestamp)

    def _run_all_nodes_before(self, timestamp):
        changed = True
        while changed:
            current_node, timestamp_before_process = self._get_next_node(timestamp)

            if timestamp_before_process <= timestamp:
                changed = True
                self.last_processed[current_node] = timestamp_before_process
                node = self.nodes[current_node]
                try:
                    node.set_time(timestamp_before_process)  # TODO how do we pass the current time to the node?
                except:
                    print(current_node + ".set_time(time) not implemented")
                node.process()
                outputs = self._pull_output(
                    current_node, timestamp_before_process)
                for port_name, sample in outputs.items():
                    self._push_input(port_name, sample)

                # TODO use output_ports to implement port trigger
                # output_ports = self.output_ports[current_node].keys()
            else:
                changed = False

    def _get_next_node(self, timestamp):
        timestamp_before_process = float("inf")
        current_node = None
        for node_name in self.last_processed.keys():
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
        for port_name, getter in self.output_ports[node_name].items():
            if port_name not in self.output_port_buffers:
                print("[DataFlowControl] type for port %s not defined"
                      % port_name)
                continue
            if self.verbose >= 1:
                print("[DataFlowControl] getting %s" % port_name)
            sample = getter()
            self._update_envire_item(sample, port_name, timestamp)
            outputs[port_name] = sample
        return outputs

    def _push_input(self, output_port, sample):
        if output_port in self.connections:
            input_port = self.connections[output_port]
            if self.verbose >= 1:
                print("[DataFlowControl] setting %s" % input_port)
            node_name, _ = input_port.split(".")
            setter = self.input_ports[node_name][input_port]
            setter(sample)
        elif self.verbose >= 2:
            print("[DataFlowControl] port '%s' is not connected"
                  % output_port)


class TextVisualization:
    """Text "visualization".

    This is just for debugging purposes.
    """
    def __init__(self):
        pass
