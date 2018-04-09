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
    graph = pydot.Dot(graph_type="digraph")

    fillcolor = "#555555"
    for node_name in dfc.nodes.keys():
        if node_name in dfc.periods:
            label = "cycle time: %.3f s" % _microseconds_to_seconds(
                dfc.periods[node_name])
        else:  # TODO port triggered
            label = ""
        cluster = pydot.Cluster(__display_name(node_name), label=label)
        component_node = pydot.Node(
            __display_name(node_name), style="filled", fillcolor=fillcolor)
        cluster.add_node(component_node)
        _add_ports_to_cluster(node_name, dfc.input_ports_[node_name], cluster,
                              output=False)
        _add_ports_to_cluster(node_name, dfc.output_ports_[node_name], cluster)
        graph.add_subgraph(cluster)

    fillcolor = "#bb5555"
    for node_name in dfc.log_ports_.keys():
        cluster = pydot.Cluster(__display_name(node_name), label="")
        component_node = pydot.Node(
            __display_name(node_name), style="filled", fillcolor=fillcolor)
        cluster.add_node(component_node)
        _add_ports_to_cluster(node_name, dfc.log_ports_[node_name], cluster)
        graph.add_subgraph(cluster)

    for node_name in dfc.result_ports_.keys():
        cluster = pydot.Cluster(__display_name(node_name), label="")
        component_node = pydot.Node(
            __display_name(node_name), style="filled", fillcolor=fillcolor)
        cluster.add_node(component_node)
        _add_ports_to_cluster(node_name, dfc.result_ports_[node_name], cluster,
                              output=False)
        graph.add_subgraph(cluster)

    for output_port, input_port in dfc.connections:
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
