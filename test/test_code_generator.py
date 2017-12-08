import os
import yaml
from cdff_dev.code_generator import write_dfn
from cdff_dev.testing import EnsureCleanup, build_extension
from nose.tools import assert_true, assert_equal


def test_square_smoke():
    with open("test/test_data/square_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/square"
    with EnsureCleanup(tmp_folder) as ec:
        filenames = write_dfn(node, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        build_extension(
            tmp_folder, hide_stderr=True,
            name="dfn_ci_" + node["name"].lower(),
            pyx_filename=os.path.join(tmp_folder, "python", "dfn_ci_" + node["name"].lower() + ".pyx"),
            implementation=map(lambda filename: os.path.join(tmp_folder, "src", filename),
                               ["Square.cpp", "SquareInterface.cpp"]),
            sourcedir=os.path.join(tmp_folder, "src"), incdirs=["test/test_output/"],
            compiler_flags=[], library_dirs=[], libraries=[],
            includes=[]
        )

        import dfn_ci_square
        square = dfn_ci_square.Square()
        assert_true(square.configure())
        square.xInput(5.0)
        assert_true(square.process())
        assert_equal(25.0, square.x_squaredOutput())
