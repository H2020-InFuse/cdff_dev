try:
    import pydot
except ImportError:
    import warnings
    warnings.warn("'pydot' is missing. It is required for "
                  "DataFlowControl.save_graph_png(filename).")


def save_graph_png(dfc, filename):
    """Creates a graph visualization of the managed processing network.

    Parameters
    ----------
    dfc : DataFlowControl
        Fully initialized data flow control

    filename : str
        Name of the output file (should have a '.png' ending)
    """
    save_network_graph_as_png(
        filename, dfc.nodes.keys(), dfc.input_ports_, dfc.output_ports_,
        dfc.log_ports_, dfc.result_ports_, dfc.connections, dfc.periods,
        dfc.trigger_ports)


def save_network_graph_as_png(
        filename, nodes, node_inputs, node_outputs, network_inputs,
        network_outputs, connections, periods=None, trigger_ports=None):
    """Save network graph as PNG.

    Parameters
    ----------
    filename : str
        Name of the PNG file

    nodes : list
        List of nodes

    node_inputs : dict
        Mapping from node names to list of input port names

    node_outputs : dict
        Mapping from node names to list of output port names

    network_inputs : dict
        Mapping from node names to list of network input names

    network_outputs : dict
        Mapping from node names to list of network output names

    connections : list
        List of pairs of input port 'node.input_port' and output port
        'node.output_port'

    periods : dict, optional (default: {})
        Mapping from node names to trigger periods

    trigger_ports : dict, optional (default: {})
        Mapping from node names to trigger ports
    """
    if periods is None:
        periods = dict()
    if trigger_ports is None:
        trigger_ports = dict()

    graph = pydot.Dot(graph_type="digraph")
    fillcolor = "#555555"

    for node_name in nodes:
        if node_name in periods:
            label = "cycle time: %.3f s" % _microseconds_to_seconds(
                periods[node_name])
        elif node_name in trigger_ports:
            label = "triggered on %s" % trigger_ports[node_name]
        else:
            label = "no trigger"
        cluster = pydot.Cluster(__display_name(node_name), label=label)
        component_node = pydot.Node(
            __display_name(node_name), style="filled", fillcolor=fillcolor)
        cluster.add_node(component_node)
        _add_ports_to_cluster(node_name, node_inputs[node_name], cluster,
                              output=False)
        _add_ports_to_cluster(node_name, node_outputs[node_name], cluster)
        graph.add_subgraph(cluster)

    fillcolor = "#bb5555"
    for node_name in network_inputs.keys():
        cluster = pydot.Cluster(__display_name(node_name), label="")
        component_node = pydot.Node(
            __display_name(node_name), style="filled", fillcolor=fillcolor)
        cluster.add_node(component_node)
        _add_ports_to_cluster(node_name, network_inputs[node_name], cluster)
        graph.add_subgraph(cluster)

    for node_name in network_outputs.keys():
        cluster = pydot.Cluster(__display_name(node_name), label="")
        component_node = pydot.Node(
            __display_name(node_name), style="filled", fillcolor=fillcolor)
        cluster.add_node(component_node)
        _add_ports_to_cluster(node_name, network_outputs[node_name], cluster,
                              output=False)
        graph.add_subgraph(cluster)

    for output_port, input_port in connections:
        connection_edge = pydot.Edge(
            __display_name(output_port), __display_name(input_port), penwidth=3)
        graph.add_edge(connection_edge)

    graph.write_png(filename)


def _add_ports_to_cluster(node_name, ports, cluster, output=True):
    for port in ports:
        port_name = node_name + "." + port
        port_node = pydot.Node(__display_name(port_name), style="filled",
                               fillcolor="#aaaaaa")
        cluster.add_node(port_node)
        a = __display_name(node_name)
        b = __display_name(port_name)
        if output:
            edge = pydot.Edge(a, b)
        else:
            edge = pydot.Edge(b, a)
        cluster.add_edge(edge)


def __display_name(element):
    return element.replace("/", "")


def _microseconds_to_seconds(microseconds):
    return float(microseconds) / 1000000.0
