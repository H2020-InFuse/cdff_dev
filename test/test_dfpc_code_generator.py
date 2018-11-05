import os
import yaml
from cdff_dev.code_generator import write_dfpc
from cdff_dev.path import load_cdffpath
import cdff_types
from cdff_dev.testing import EnsureCleanup, build_extension
from nose.tools import assert_true, assert_raises_regexp


hide_stderr = True


def test_generate_files():
    with open("test/test_data/pointcloud_generation_dfpc_desc.yml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/dfpc_ci_PointcloudGeneration_test_files"
    with EnsureCleanup(tmp_folder) as ec:
        cdffpath = load_cdffpath()
        filenames = write_dfpc(node, cdffpath, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        assert_true(os.path.exists(os.path.join(
            tmp_folder, "PointcloudGenerationInterface.hpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "PointcloudGenerationInterface.cpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "PointcloudGenerationImplementation.hpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "PointcloudGenerationImplementation.cpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "python", "_pointcloudgeneration.pxd")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "python", "pointcloudgeneration.pxd")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "python", "pointcloudgeneration.pyx")))


def test_compile():
    with open("test/test_data/pointcloud_generation_dfpc_desc.yml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/dfpc_ci_PointcloudGeneration"
    with EnsureCleanup(tmp_folder) as ec:
        cdffpath = load_cdffpath()
        filenames = write_dfpc(node, cdffpath, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        incdirs = ["test/test_output/", os.path.join(cdffpath, "Common"),
                   os.path.join(cdffpath, "DFPCs")]
        build_extension(
            tmp_folder, hide_stderr=hide_stderr,
            name=node["name"].lower(),
            pyx_filename=os.path.join(
                tmp_folder, "python", node["name"].lower() + ".pyx"),
            implementation=map(
                lambda filename: os.path.join(tmp_folder, filename),
                ["PointcloudGenerationImplementation.cpp",
                 "PointcloudGenerationInterface.cpp"]),
            sourcedir=tmp_folder, incdirs=incdirs,
            compiler_flags=[], library_dirs=[], libraries=[],
            includes=[]
        )

        from pointcloudgeneration import PointcloudGenerationImplementation

        pg = PointcloudGenerationImplementation()

        pg.setup()

        l = cdff_types.LaserScan()
        pg.lidar2DInput(l)
        r = cdff_types.RigidBodyState()
        pg.dyamixelDynamicTfInput(r)

        pg.run()

        is_available = pg.pointcloudAvailableOutput()
        p = pg.getPointcloud(30)


def test_forbidden_type():
    with open("test/test_data/forbidden_type_dfpc_desc.yml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/dfpc_ci_ForbiddenTypeDFPC"
    with EnsureCleanup(tmp_folder) as ec:
        cdffpath = load_cdffpath()
        assert_raises_regexp(
            TypeError,
            "Type 'asn1SccVisualPointFeatureVector2D' is not allowed.",
            write_dfpc, node, cdffpath, tmp_folder)
