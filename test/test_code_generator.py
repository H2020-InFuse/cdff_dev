import yaml
from cdff_dev.code_generator import write_dfn


def test_square_smoke():
    with open("test/test_data/square_desc.yaml", "r") as f:
        node = yaml.load(f)
    write_dfn(node, "test/test_data")
