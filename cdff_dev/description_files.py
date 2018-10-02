class DFNDescriptionException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


class DFPCDescriptionException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


class PortDescriptionException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


def validate_node(node):
    """Validate node description.

    Raises a DFNDescriptionException if validation is not
    successful.

    A validated node contains:
    - name
    - input_ports
    - output_ports
    - implementations

    Parameters
    ----------
    node : dict
        Node description

    Returns
    -------
    validated_node : dict
        Validated node description
    """
    validated_node = {}
    validated_node.update(node)

    if "name" not in node:
        raise DFNDescriptionException(
            "DFN description has no attribute 'name'.")

    _validate_ports(validated_node)

    if "implementations" not in node:
        validated_node["implementations"] = [node["name"]]

    return validated_node


def validate_dfpc(dfpc):
    """Validate DFPC description.

    Raises a DFNDescriptionException if validation is not
    successful.

    A validated DFPC contains:
    - name
    - input_ports
    - output_ports
    - implementations
        - list of DFNs
        - connections between inputs / outputs and DFNs
        - internal connections between DFNs

    Parameters
    ----------
    dfpc : dict
        DFPC description

    Returns
    -------
    validated_dfpc : dict
        Validated DFPC description
    """
    validated_dfpc = {}
    validated_dfpc.update(dfpc)

    # TODO this has become really complicated, maybe we should use
    #      pykwalify: https://pypi.org/project/pykwalify/

    if "name" not in dfpc:
        raise DFPCDescriptionException(
            "DFPC description has no attribute 'name'.")

    _validate_doc(validated_dfpc)
    _validate_ports(validated_dfpc)
    _validate_dfpc_operations(validated_dfpc)
    _validate_implementations(validated_dfpc)

    return validated_dfpc


def _validate_doc(desc):
    if "doc" not in desc:
        raise DFPCDescriptionException("Entry 'doc' for DFPC is missing.")


def _validate_ports(desc):
    if "input_ports" not in desc:
        desc["input_ports"] = []

    if "output_ports" not in desc:
        desc["output_ports"] = []

    for port in desc["input_ports"] + desc["output_ports"]:
        if "name" not in port:
            raise PortDescriptionException("Port has no name: %s" % port)
        if "type" not in port:
            raise PortDescriptionException("Port has no type: %s" % port)


def _validate_dfpc_operations(desc):
    if "operations" not in desc:
        desc["operations"] = []
        return

    for op in desc["operations"]:
        if "name" not in op:
            raise DFPCDescriptionException("Operation has no name: %s" % op)
        if "output_type" not in op:
            op["output_type"] = "void"

        if "inputs" not in op:
            op["inputs"] = []
        for inp in op["inputs"]:
            if "name" not in inp:
                raise DFPCDescriptionException(
                    "Input of operation '%s' has no name: %s"
                    % (op["name"], inp))
            if "type" not in inp:
                raise DFPCDescriptionException(
                    "Input '%s' of operation '%s' has no type"
                    % (inp["name"], op["name"]))


def _validate_implementations(desc):
    if "implementations" not in desc:
        desc["implementations"] = [
            {
                "name": desc["name"],
                "doc": desc["doc"]
            }
        ]

    for implementation in desc["implementations"]:
        if "name" not in implementation:
            raise DFPCDescriptionException("Implementation has no name")
        dfn_ids = _validate_dfns(implementation)
        _validate_dfpc_input_connections(implementation, dfn_ids)
        _validate_dfpc_output_connections(implementation, dfn_ids)
        _validate_dfpc_internal_connections(implementation, dfn_ids)


def _validate_dfns(implementation):
    if "dfns" not in implementation:
        implementation["dfns"] = []
    dfn_ids = []
    for dfn in implementation["dfns"]:
        if "dfn_id" not in dfn:
            raise DFPCDescriptionException("dfn_id is missing")
        if "name" not in dfn:
            raise DFPCDescriptionException(
                "Name of DFN '%s' is missing" % dfn["dfn_id"])
        if "implementation" not in dfn:
            raise DFPCDescriptionException(
                "Implementation of DFN '%s' is missing" % dfn["dfn_id"])
        if "activation" not in dfn:
            raise DFPCDescriptionException(
                "Activation of DFN '%s' is missing" % dfn["dfn_id"])
        _validate_dfn_activation(dfn["activation"], dfn["dfn_id"])
        dfn_ids.append(dfn["dfn_id"])
    return dfn_ids


def _validate_dfn_activation(activation, dfn_id):
    if "type" not in activation:
        raise DFPCDescriptionException(
            "No activation type for dfn_id '%s'" % dfn_id)
    if activation["type"] not in ["input_triggered", "frequency"]:
        raise DFPCDescriptionException(
            "Unknown activation type for dfn_id '%s': '%s'. Only "
            "'input_triggered' and 'frequency' are allowed"
            % (dfn_id, activation["type"]))
    if "value" not in activation:
        raise DFPCDescriptionException(
            "No activation value for dfn_id '%s'" % dfn_id)


def _validate_dfpc_input_connections(implementation, dfn_ids):
    if "input_connections" not in implementation:
        implementation["input_connections"] = []
    for input_connection in implementation["input_connections"]:
        if "dfpc_input" not in input_connection:
            raise DFPCDescriptionException(
                "Missing dfpc_input in input connection")
        # TODO test if dfpc_input corresponds to a previously defined input port
        if "dfn_id" not in input_connection:
            raise DFPCDescriptionException(
                "Missing dfn_id in input connection '%s'"
                % input_connection["dfpc_input"])
        if input_connection["dfn_id"] not in dfn_ids:
            raise DFPCDescriptionException(
                "dfn_id '%s' in input connection '%s' not defined"
                % (input_connection["dfn_id"],
                   input_connection["dfpc_input"]))
        if "port" not in input_connection:
            raise DFPCDescriptionException(
                "Missing port in input connection '%s'"
                % input_connection["dfpc_input"])


def _validate_dfpc_output_connections(implementation, dfn_ids):
    if "output_connections" not in implementation:
        implementation["output_connections"] = []
    for output_connection in implementation["output_connections"]:
        if "dfpc_output" not in output_connection:
            raise DFPCDescriptionException(
                "Missing dfpc_output in output connection")
        # TODO test if dfpc_output corresponds to a previously defined output port
        if "dfn_id" not in output_connection:
            raise DFPCDescriptionException(
                "Missing dfn_id in output connection '%s'"
                % output_connection["dfpc_output"])
        if output_connection["dfn_id"] not in dfn_ids:
            raise DFPCDescriptionException(
                "dfn_id '%s' in output connection '%s' not defined"
                % (output_connection["dfn_id"],
                   output_connection["dfpc_output"]))
        if "port" not in output_connection:
            raise DFPCDescriptionException(
                "Missing port in output connection '%s'"
                % output_connection["dfpc_output"])


def _validate_dfpc_internal_connections(implementation, dfn_ids):
    if "internal_connections" not in implementation:
        implementation["internal_connections"] = []

    for conn in implementation["internal_connections"]:
        if "from" not in conn:
            raise DFPCDescriptionException(
                "No 'from' in internal connection: %s" % conn)
        if "to" not in conn:
            raise DFPCDescriptionException(
                "No 'to' in internal connection: %s" % conn)

        for port in [conn["from"], conn["to"]]:
            if "dfn_id" not in port:
                raise DFPCDescriptionException(
                    "No DFN is specified in internal connection: %s" % conn)
            if port["dfn_id"] not in dfn_ids:
                raise DFPCDescriptionException(
                    "dfn_id '%s' in internal connection not defined"
                    % port["dfn_id"])
            if "port" not in port:
                raise DFPCDescriptionException(
                    "No port is specified in internal connection: %s" % conn)