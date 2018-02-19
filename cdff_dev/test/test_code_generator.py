from cdff_dev.code_generator import validate, DFNDescriptionException
from nose.tools import assert_raises_regexp, assert_in


def test_validate_missing_name():
    node = {}
    assert_raises_regexp(
        DFNDescriptionException, "no attribute.*'name'",
        validate, node)


def test_validate_adds_empty_ports():
    node = {"name": "Dummy"}
    validated_node = validate(node)
    assert_in("input_ports", validated_node)
    assert_in("output_ports", validated_node)


def test_validate_no_implementation():
    node = {"name": "Dummy"}
    validated_node = validate(node)
    assert_in("implementations", validated_node)
    assert_in("Dummy", validated_node["implementations"])


def test_validate_implementation():
    node = {"name": "Dummy", "implementations": ["DummyImpl"]}
    validated_node = validate(node)
    assert_in("implementations", validated_node)
    assert_in("DummyImpl", validated_node["implementations"])
