from cdff_dev.description_files import validate_node, DFNDescriptionException, \
    PortDescriptionException
from nose.tools import assert_raises_regexp, assert_in


def test_validate_missing_name():
    node = {}
    assert_raises_regexp(
        DFNDescriptionException, "no attribute.*'name'",
        validate_node, node)


def test_validate_adds_empty_ports():
    node = {"name": "Dummy"}
    validated_node = validate_node(node)
    assert_in("input_ports", validated_node)
    assert_in("output_ports", validated_node)


def test_validate_no_implementation():
    node = {"name": "Dummy"}
    validated_node = validate_node(node)
    assert_in("implementations", validated_node)
    assert_in("Dummy", validated_node["implementations"])


def test_validate_implementation():
    node = {"name": "Dummy", "implementations": ["DummyImpl"]}
    validated_node = validate_node(node)
    assert_in("implementations", validated_node)
    assert_in("DummyImpl", validated_node["implementations"])


def test_validate_missing_port_name():
    node = {"name": "Dummy",
            "input_ports": [
                {"type": "double"}
            ]}
    assert_raises_regexp(
        PortDescriptionException, "Port has no name",
        validate_node, node)


def test_validate_missing_port_type():
    node = {"name": "Dummy",
            "input_ports": [
                {"name": "port1"}
            ]}
    assert_raises_regexp(
        PortDescriptionException, "Port has no type",
        validate_node, node)


def test_validate_missing_port_doc():
    node = {"name": "Dummy",
            "input_ports": [
                {"name": "port1", "type": "double"}
            ]}
    assert_raises_regexp(
        PortDescriptionException, "Port has no doc",
        validate_node, node)
