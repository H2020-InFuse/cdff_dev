from cdff_dev.code_generator import (
    validate_dfpc, DFPCDescriptionException, PortDescriptionException)
from nose.tools import assert_raises_regex, assert_in, assert_equal


def test_validate_missing_name():
    dfpc = {}
    assert_raises_regex(
        DFPCDescriptionException, "no attribute.*'name'",
        validate_dfpc, dfpc)


def test_validate_adds_empty_ports():
    dfpc = {"name": "Slam"}
    validated_dfpc = validate_dfpc(dfpc)
    assert_in("input_ports", validated_dfpc)
    assert_in("output_ports", validated_dfpc)


def test_validate_missing_port_connections():
    dfpc = {"name": "Dummy",
            "input_ports": [
                {"name": "port1",
                 "type": "double"}
            ]}
    assert_raises_regex(
        PortDescriptionException, "Port has no connections",
        validate_dfpc, dfpc)


def test_validate_missing_port_one_connection():
    dfpc = {"name": "Dummy",
            "input_ports": [
                {"name": "port1",
                 "type": "double",
                 "connections": []}
            ]}
    assert_raises_regex(
        PortDescriptionException, "Port has no connections",
        validate_dfpc, dfpc)


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
