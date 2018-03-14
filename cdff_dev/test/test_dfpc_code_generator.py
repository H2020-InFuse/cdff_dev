from cdff_dev.code_generator import (
    validate_dfpc, DFPCDescriptionException, PortDescriptionException)
from nose.tools import assert_raises_regex, assert_in


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
