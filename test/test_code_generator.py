import os
import yaml
import warnings
from cdff_dev.code_generator import write_dfn
from cdff_dev.testing import EnsureCleanup, build_extension
from nose.tools import assert_true, assert_equal, assert_raises_regex, \
    assert_true


def test_generate_files():
    with open("test/test_data/square_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/square"
    with EnsureCleanup(tmp_folder) as ec:
        filenames = write_dfn(node, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        assert_true(
            os.path.exists(os.path.join(tmp_folder, "SquareInterface.hpp")))
        assert_true(
            os.path.exists(os.path.join(tmp_folder, "SquareInterface.cpp")))
        assert_true(os.path.exists(os.path.join(tmp_folder, "Square.hpp")))
        assert_true(os.path.exists(os.path.join(tmp_folder, "Square.cpp")))
        assert_true(os.path.exists(os.path.join(tmp_folder, "python",
                                                "dfn_ci_square.pxd")))
        assert_true(os.path.exists(os.path.join(tmp_folder, "python",
                                                "dfn_ci_square.pyx")))
        assert_true(os.path.exists(os.path.join(tmp_folder, "python",
                                                "_dfn_ci_square.pxd")))


def test_square():
    with open("test/test_data/square_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/square"
    with EnsureCleanup(tmp_folder) as ec:
        filenames = write_dfn(node, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        incdirs = ["test/test_output/", "CDFF/DFNs"]
        pyx_filename = os.path.join(
            tmp_folder, "python", "dfn_ci_" + node["name"].lower() + ".pyx")
        implementation = map(
            lambda filename: os.path.join(tmp_folder, filename),
            ["Square.cpp", "SquareInterface.cpp"])
        build_extension(
            tmp_folder, hide_stderr=True,
            name="dfn_ci_" + node["name"].lower(),
            pyx_filename=pyx_filename, implementation=implementation,
            sourcedir=tmp_folder, incdirs=incdirs,
            compiler_flags=[], library_dirs=[], libraries=[],
            includes=[]
        )

        import dfn_ci_square
        square = dfn_ci_square.Square()
        square.configure()
        square.set_configuration_file("")
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
            implementation=map(lambda filename: os.path.join(tmp_folder, filename),
                               ["UnknownType.cpp", "UnknownTypeInterface.cpp"]),
            sourcedir=tmp_folder, incdirs=incdirs,
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
            implementation=map(lambda filename: os.path.join(tmp_folder, filename),
                               ["Iterative.cpp", "Recursive.cpp",
                                "FactorialInterface.cpp"]),
            sourcedir=tmp_folder, incdirs=incdirs,
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

"""
def test_asn1():
    with open("test/test_data/dummy_asn1_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/asn1"
        build_extension(
            tmp_folder, hide_stderr=False,
            name="dfn_ci_" + node["name"].lower(),
            pyx_filename=os.path.join(tmp_folder, "python", "dfn_ci_" + node["name"].lower() + ".pyx"),
            implementation=map(lambda filename: os.path.join(tmp_folder, "src", filename),
                               ["Asn1Test.cpp", "Asn1TestInterface.cpp"]),
            sourcedir=os.path.join(tmp_folder, "src"), incdirs=["test/test_output/"],
        import dfn_ci_asn1test
        asn1_test = dfn_ci_asn1test.Asn1Test()
        assert_true(asn1_test.configure())
        asn1_test.currentTimeInput(5.0)
        assert_true(asn1_test.process())
        assert_equal(6.0, asn1_test.nextTimeOutput())
"""
