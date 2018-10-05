from cdff_dev.description_files import validate_dfpc, DFPCDescriptionException
from nose.tools import assert_raises_regex, assert_in, assert_equal
from copy import deepcopy
import yaml


FULLDFPC = {
    "name": "Dummy",
    "doc": "Dummy DFPC",
    "input_ports": [
        {
            "name": "a",
            "type": "A",
            "doc": "an input"
        }
    ],
    "output_ports": [
        {
            "name": "b",
            "type": "B",
            "doc": "an output"
        }
    ],
    "operations": [
        {
            "name": "c",
            "doc": "an operation",
            "inputs": [
                {
                    "name": "d",
                    "type": "D"
                }
            ],
            "output_type": "E"
        }
    ],
    "implementations": [
        {
            "name": "DummyA",
            "dfns": [
                {
                    "dfn_id": "dummyDFN1",
                    "type": "DummyDFN",
                    "implementation": "MyDummyDFN",
                    "activation": {
                        "type": "input_triggered",
                        "value": "bla"
                    }
                },
                {
                    "dfn_id": "dummyDFN2",
                    "type": "DummyDFN",
                    "implementation": "MyDummyDFN",
                    "activation": {
                        "type": "input_triggered",
                        "value": "bla"
                    }
                }
            ],
            "input_connections": [
                {
                    "dfpc_input": "bla",
                    "dfn_id": "dummyDFN1",
                    "port": "bla"
                },
            ],
            "output_connections": [
                {
                    "dfpc_output": "blub",
                    "dfn_id": "dummyDFN2",
                    "port": "blub"
                }
            ],
            "internal_connections": [
                {
                    "from": {
                        "dfn_id": "dummyDFN1",
                        "port": "blub"
                    },
                    "to": {
                        "dfn_id": "dummyDFN2",
                        "port": "bla"
                    }
                }
            ]
        }
    ],
}


def test_validate_complete_example():
    validate_dfpc(FULLDFPC)


def test_validate_missing_name():
    dfpc = {}
    assert_raises_regex(
        DFPCDescriptionException, "no attribute.*'name'",
        validate_dfpc, dfpc)


def test_validate_missing_doc():
    dfpc = {"name": "Slam"}
    validated_dfpc = validate_dfpc(dfpc)
    assert_in("doc", validated_dfpc)


def test_validate_adds_empty_ports():
    dfpc = {"name": "Slam"}
    validated_dfpc = validate_dfpc(dfpc)
    assert_in("input_ports", validated_dfpc)
    assert_in("output_ports", validated_dfpc)


def test_validate_missing_operations():
    dfpc = {"name": "Dummy"}
    validated_dfpc = validate_dfpc(dfpc)
    assert_in("operations", validated_dfpc)


def test_validate_missing_operation_name():
    dfpc = {"name": "Dummy",
            "operations": [
                {"inputs": [
                     {"name": "bla",
                      "type": "double"}
                 ],
                 "output_type": "double"
                }
            ]}
    assert_raises_regex(
        DFPCDescriptionException, "Operation has no name",
        validate_dfpc, dfpc)


def test_validate_missing_output_type_is_void():
    dfpc = {"name": "Dummy",
            "operations": [
                {"name": "dummy",
                 "inputs": [
                     {"name": "bla",
                      "type": "double"}
                 ]
                }
            ]}
    validated_dfpc = validate_dfpc(dfpc)
    assert_in("output_type", validated_dfpc["operations"][0])
    assert_equal(validated_dfpc["operations"][0]["output_type"], "void")


def test_validate_missing_input_name():
    dfpc = {"name": "Dummy",
            "operations": [
                {"name": "dummy",
                 "inputs": [
                     {"type": "double"}
                 ]
                }
            ]}
    assert_raises_regex(
        DFPCDescriptionException,
        "Input of operation 'dummy' has no name", validate_dfpc, dfpc)


def test_validate_missing_input_type():
    dfpc = {"name": "Dummy",
            "operations": [
                {"name": "dummy",
                 "inputs": [
                     {"name": "bla"}
                 ]
                }
            ]}
    assert_raises_regex(
        DFPCDescriptionException,
        "Input 'bla' of operation 'dummy' has no type", validate_dfpc, dfpc)


def test_validate_missing_implementation_name():
    dfpc = {"name": "Dummy", "implementations": [{}]}
    assert_raises_regex(
        DFPCDescriptionException,
        "Implementation has no name", validate_dfpc, dfpc)


def test_validate_missing_dfn_list():
    dfpc = {
        "name": "Dummy",
        "implementations": [
            {"name": "DummyImpl"}
        ]
    }
    validated_dfpc = validate_dfpc(dfpc)
    assert_in("dfns", validated_dfpc["implementations"][0])


def test_validate_missing_dfn_id():
    dfpc = {
        "name": "Dummy",
        "implementations": [
            {"name": "DummyImpl",
             "dfns": [{}]}
        ]
    }
    assert_raises_regex(
        DFPCDescriptionException, "dfn_id is missing", validate_dfpc, dfpc)


def test_validate_missing_dfn_type():
    dfpc = {
        "name": "Dummy",
        "implementations": [
            {"name": "DummyImpl",
             "dfns": [
                 {
                     "dfn_id": "dummy"
                 }
             ]}
        ]
    }
    assert_raises_regex(
        DFPCDescriptionException, "Type of DFN 'dummy' is missing",
        validate_dfpc, dfpc)


def test_validate_missing_dfn_implementation():
    dfpc = {
        "name": "Dummy",
        "implementations": [
            {"name": "DummyImpl",
             "dfns": [
                 {
                     "dfn_id": "dummy",
                     "type": "DummyDFN"
                 }
             ]}
        ]
    }
    assert_raises_regex(
        DFPCDescriptionException, "Implementation of DFN 'dummy' is missing",
        validate_dfpc, dfpc)


def test_validate_missing_dfn_activation():
    dfpc = {
        "name": "Dummy",
        "implementations": [
            {"name": "DummyImpl",
             "dfns": [
                 {
                     "dfn_id": "dummy",
                     "type": "DummyDFN",
                     "implementation": "A"
                 }
             ]}
        ]
    }
    assert_raises_regex(
        DFPCDescriptionException, "Activation of DFN 'dummy' is missing",
        validate_dfpc, dfpc)


def test_validate_missing_dfn_activation_type():
    dfpc = {
        "name": "Dummy",
        "implementations": [
            {"name": "DummyImpl",
             "dfns": [
                 {
                     "dfn_id": "dummy",
                     "type": "DummyDFN",
                     "implementation": "A",
                     "activation": {
                         "value": "bla"
                     }
                 }
             ]}
        ]
    }
    assert_raises_regex(
        DFPCDescriptionException, "No activation type for dfn_id 'dummy'",
        validate_dfpc, dfpc)


def test_validate_unknown_dfn_activation_type():
    dfpc = {
        "name": "Dummy",
        "implementations": [
            {"name": "DummyImpl",
             "dfns": [
                 {
                     "dfn_id": "dummy",
                     "type": "DummyDFN",
                     "implementation": "A",
                     "activation": {
                         "type": "murks",
                         "value": "bla"
                     }
                 }
             ]}
        ]
    }
    assert_raises_regex(
        DFPCDescriptionException,
        "Unknown activation type for dfn_id 'dummy': 'murks'. Only "
        "'input_triggered' and 'frequency' are allowed",
        validate_dfpc, dfpc)


def test_validate_missing_dfn_activation_value():
    dfpc = {
        "name": "Dummy",
        "implementations": [
            {"name": "DummyImpl",
             "dfns": [
                 {
                     "dfn_id": "dummy",
                     "type": "DummyDFN",
                     "implementation": "A",
                     "activation": {
                         "type": "input_triggered"
                     }
                 }
             ]}
        ]
    }
    assert_raises_regex(
        DFPCDescriptionException, "No activation value for dfn_id 'dummy'",
        validate_dfpc, dfpc)


def test_validate_missing_dfn_input_connections():
    dfpc = {
        "name": "Dummy",
        "implementations": [{"name": "DummyImpl"}]
    }
    validated_dfpc = validate_dfpc(dfpc)
    assert_in("input_connections", validated_dfpc["implementations"][0])


def test_validate_missing_dfpc_input_in_input_connections():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["input_connections"][0]["dfpc_input"]
    assert_raises_regex(
        DFPCDescriptionException, "Missing dfpc_input in input connection",
        validate_dfpc, dfpc)


def test_validate_missing_dfn_id_in_input_connections():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["input_connections"][0]["dfn_id"]
    assert_raises_regex(
        DFPCDescriptionException, "Missing dfn_id in input connection 'bla'",
        validate_dfpc, dfpc)


def test_validate_unknown_dfn_id_in_input_connections():
    dfpc = deepcopy(FULLDFPC)
    dfpc["implementations"][0]["input_connections"][0]["dfn_id"] = "?"
    assert_raises_regex(
        DFPCDescriptionException,
        "dfn_id '\?' in input connection 'bla' not defined",
        validate_dfpc, dfpc)


def test_validate_missing_port_in_input_connections():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["input_connections"][0]["port"]
    assert_raises_regex(
        DFPCDescriptionException, "Missing port in input connection 'bla'",
        validate_dfpc, dfpc)


def test_validate_missing_dfn_output_connections():
    dfpc = {
        "name": "Dummy",
        "implementations": [{"name": "DummyImpl"}]
    }
    validated_dfpc = validate_dfpc(dfpc)
    assert_in("output_connections", validated_dfpc["implementations"][0])


def test_validate_missing_dfpc_output_in_output_connections():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["output_connections"][0]["dfpc_output"]
    assert_raises_regex(
        DFPCDescriptionException, "Missing dfpc_output in output connection",
        validate_dfpc, dfpc)


def test_validate_missing_dfn_id_in_output_connections():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["output_connections"][0]["dfn_id"]
    assert_raises_regex(
        DFPCDescriptionException, "Missing dfn_id in output connection 'blub'",
        validate_dfpc, dfpc)


def test_validate_unknown_dfn_id_in_output_connections():
    dfpc = deepcopy(FULLDFPC)
    dfpc["implementations"][0]["output_connections"][0]["dfn_id"] = "?"
    assert_raises_regex(
        DFPCDescriptionException,
        "dfn_id '\?' in output connection 'blub' not defined",
        validate_dfpc, dfpc)


def test_validate_missing_port_in_output_connections():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["output_connections"][0]["port"]
    assert_raises_regex(
        DFPCDescriptionException, "Missing port in output connection 'blub'",
        validate_dfpc, dfpc)


def test_validate_connection_missing_from():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["internal_connections"][0]["from"]
    assert_raises_regex(
        DFPCDescriptionException,
        "No 'from' in internal connection", validate_dfpc, dfpc)


def test_validate_connection_missing_to():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["internal_connections"][0]["to"]
    assert_raises_regex(
        DFPCDescriptionException,
        "No 'to' in internal connection", validate_dfpc, dfpc)


def test_validate_connection_missing_dfn_id():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["internal_connections"][0]["to"]["dfn_id"]
    assert_raises_regex(
        DFPCDescriptionException,
        "No DFN is specified in internal connection", validate_dfpc, dfpc)


def test_validate_connection_unknown_dfn_id():
    dfpc = deepcopy(FULLDFPC)
    dfpc["implementations"][0]["internal_connections"][0]["to"]["dfn_id"] = "A"
    assert_raises_regex(
        DFPCDescriptionException,
        "dfn_id 'A' in internal connection not defined",
        validate_dfpc, dfpc)


def test_validate_connection_missing_port():
    dfpc = deepcopy(FULLDFPC)
    del dfpc["implementations"][0]["internal_connections"][0]["to"]["port"]
    assert_raises_regex(
        DFPCDescriptionException,
        "No port is specified in internal connection", validate_dfpc, dfpc)


def test_validate_smoke_test():
    with open("test/test_data/pointcloud_generation_dfpc_desc.yml") as f:
        desc = yaml.load(f)
    validate_dfpc(desc)
