import os
import yaml
import warnings
from cdff_dev.code_generator import write_dfpc
from cdff_dev.testing import EnsureCleanup, build_extension
from nose.tools import assert_true, assert_equal, assert_raises_regex, \
    assert_true


def test_generate_files():
    with open("test/test_data/LidarSlam_dfpc_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/dfpc_ci_lidarSlam"
    with EnsureCleanup(tmp_folder) as ec:
        filenames = write_dfpc(node, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        assert_true(os.path.exists(os.path.join(
            tmp_folder, "LidarSlamInterface.hpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "LidarSlamInterface.cpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "LidarSlamRock.hpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "LidarSlamRock.cpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "python", "dfpc_ci_lidarslam.pxd")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "python", "dfpc_ci_lidarslam.pyx")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "python", "_dfpc_ci_lidarslam.pxd")))
