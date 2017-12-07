import os
import yaml
from cdff_dev.code_generator import write_dfn
from cdff_dev.testing import ensure_cleanup, build_extension


def test_square_smoke():
    with open("test/test_data/square_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = ".test_tmp"
    with open("test/test_data/square_desc.yaml", "r") as f:#with ensure_cleanup(tmp_folder):
        write_dfn(node, tmp_folder)
        # TODO refactor
        build_extension(
            tmp_folder, name="dfn_ci_" + node["name"].lower(),
            pyx_filename=os.path.join(tmp_folder, "python", "dfn_ci_" + node["name"].lower() + ".pyx"),
            implementation=map(lambda filename: os.path.join(tmp_folder, "src", filename),
                               ["Square.cpp", "SquareInterface.cpp"]),
            sourcedir=os.path.join(tmp_folder, "src"), incdirs=["."],
            compiler_flags=[], library_dirs=[], libraries=[])
