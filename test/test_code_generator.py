import os
import yaml
import warnings
from cdff_dev.code_generator import write_dfn
from cdff_dev.testing import EnsureCleanup, build_extension
from nose.tools import assert_true, assert_equal, assert_raises_regex


def test_square_smoke():
    with open("test/test_data/square_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/square"
    with EnsureCleanup(tmp_folder) as ec:
        filenames = write_dfn(node, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        incdirs = ["test/test_output/", "CDFF/DFNs"]
        build_extension(
            tmp_folder, hide_stderr=True,
            name="dfn_ci_" + node["name"].lower(),
            pyx_filename=os.path.join(tmp_folder, "python", "dfn_ci_" + node["name"].lower() + ".pyx"),
            implementation=map(lambda filename: os.path.join(tmp_folder, "src", filename),
                               ["Square.cpp", "SquareInterface.cpp"]),
            sourcedir=os.path.join(tmp_folder, "src"), incdirs=incdirs,
            compiler_flags=[], library_dirs=[], libraries=[],
            includes=[]
        )

        import dfn_ci_square
        square = dfn_ci_square.Square()
        square.configure()
        square.xInput(5.0)
        square.process()
        assert_equal(25.0, square.x_squaredOutput())


def test_unknown_type():
    with open("test/test_data/unknowntype_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/unknowntype"
    with EnsureCleanup(tmp_folder) as ec:
        warnings.simplefilter("ignore", UserWarning)
        filenames = write_dfn(node, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        incdirs = ["test/test_output/", "CDFF/DFNs"]
        assert_raises_regex(
            Exception, "Exit status",
            build_extension,
            tmp_folder, hide_stderr=True,
            name="dfn_ci_" + node["name"].lower(),
            pyx_filename=os.path.join(tmp_folder, "python", "dfn_ci_" + node["name"].lower() + ".pyx"),
            implementation=map(lambda filename: os.path.join(tmp_folder, "src", filename),
                               ["UnknownType.cpp", "UnknownTypeInterface.cpp"]),
            sourcedir=os.path.join(tmp_folder, "src"), incdirs=incdirs,
            compiler_flags=[], library_dirs=[], libraries=[],
            includes=[]
        )


def test_multiple_implementations():
    with open("test/test_data/multiple_implementations_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/multiple_implementations"
    with EnsureCleanup(tmp_folder) as ec:
        filenames = write_dfn(node, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        incdirs = ["test/test_output/", "CDFF/DFNs"]
        build_extension(
            tmp_folder, hide_stderr=True,
            name="dfn_ci_" + node["name"].lower(),
            pyx_filename=os.path.join(
                tmp_folder, "python", "dfn_ci_" + node["name"].lower() + ".pyx"),
            implementation=map(lambda filename: os.path.join(tmp_folder, "src", filename),
                               ["Iterative.cpp", "Recursive.cpp",
                                "FactorialInterface.cpp"]),
            sourcedir=os.path.join(tmp_folder, "src"), incdirs=incdirs,
            compiler_flags=[], library_dirs=[], libraries=[],
            includes=[]
        )

        import dfn_ci_factorial
        square = dfn_ci_factorial.Iterative()
        square.configure()
        square.xInput(5.0)
        square.process()

        square = dfn_ci_factorial.Recursive()
        square.configure()
        square.xInput(5.0)
        square.process()
