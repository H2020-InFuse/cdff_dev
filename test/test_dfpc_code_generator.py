import os
import yaml
from cdff_dev.code_generator import write_dfpc
from cdff_dev.path import load_cdffpath, CTYPESDIR
import cdff_types
from cdff_dev.testing import EnsureCleanup, build_extension
from nose.tools import assert_true


def test_generate_files():
    with open("test/test_data/pointcloud_generation_dfpc_desc.yml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/dfpc_ci_PointcloudGeneration"
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
            tmp_folder, "PointcloudGeneration.hpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "PointcloudGeneration.cpp")))


def test_compile():
    with open("test/test_data/pointcloud_generation_dfpc_desc.yml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/dfpc_ci_PointcloudGeneration"
    with EnsureCleanup(tmp_folder) as ec:
        cdffpath = load_cdffpath()
        filenames = write_dfpc(node, cdffpath, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        incdirs = ["test/test_output/", os.path.join(cdffpath, "DFPCs"),
                   os.path.join(cdffpath, CTYPESDIR)]
        build_extension(
            tmp_folder, hide_stderr=False,
            name=node["name"].lower(),
            pyx_filename=os.path.join(
                tmp_folder, "python", node["name"].lower() + ".pyx"),
            implementation=map(
                lambda filename: os.path.join(tmp_folder, filename),
                ["PointcloudGeneration.cpp",
                 "PointcloudGenerationInterface.cpp"]),
            sourcedir=tmp_folder, incdirs=incdirs,
            compiler_flags=[], library_dirs=[], libraries=[],
            includes=[]
        )

        from pointcloudgeneration import PointcloudGeneration

        pg = PointcloudGeneration()

        pg.setup()

        l = cdff_types.LaserScan()
        pg.lidar2DInput(l)
        r = cdff_types.RigidBodyState()
        pg.dyamixelDynamicTfInput(r)

        pg.run()

        p = pg.pointcloudOutput()
        p = pg.getPointcloud(30)
