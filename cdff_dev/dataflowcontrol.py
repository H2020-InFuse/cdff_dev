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

    visualization : Visualization, optional (default: None)
        Visualization

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(self, nodes, connections, visualization=None, verbose=0):
        self.nodes = nodes
        self.connections = connections
        self.visualization = visualization
        self.verbose = verbose

    def setup(self):
        self._configure_nodes()
        self.port_cache = self._cache_ports()

    def _configure_nodes(self):
        for node in self.nodes.values():
            node.configure()

    def _cache_ports(self):
        ports = dict()
        for output_port, input_port in self.connections:
            ports[output_port] = self._check_port(output_port, "Output")
            ports[input_port] = self._check_port(input_port, "Input")
        return ports

    def _check_port(self, port, port_type):
        if port_type not in ["Input", "Output"]:
            raise ValueError("port_type must be either 'Input' or 'Output'")

        node_name, port_name = port.split(".")
        if node_name not in self.nodes:
            if port_type == "Input":
                raise ValueError("Unknown node '%s' from port name '%s'."
                                 % (node_name, port))
            else:
                return None

        node = self.nodes[node_name]
        method_name = port_name + port_type

        if port_type == "Input" and not hasattr(node, method_name):
            raise AttributeError(
                "Could not find %s.%s(...) corresponding to port %s"
                % (type(node), method_name, port_name))

        return getattr(node, method_name)


class TextVisualization:
    """Text "visualization".

    This is just for debugging purposes.
    """
    def __init__(self):
        pass
