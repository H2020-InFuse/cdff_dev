from cdff_dev.code_generator import validate_dfpc, DFPCDescriptionException
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
