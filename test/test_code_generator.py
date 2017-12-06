import yaml
from cdff_dev.code_generator import write_dfn
from cdff_dev.testing import ensure_cleanup


def test_square_smoke():
    with open("test/test_data/square_desc.yaml", "r") as f:
        node = yaml.load(f)
    with ensure_cleanup(".test_tmp"):
        write_dfn(node, ".test_tmp")
