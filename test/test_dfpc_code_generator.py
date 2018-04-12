import os
import yaml
import warnings
from cdff_dev.code_generator import write_dfpc
from cdff_dev.testing import EnsureCleanup, build_extension
from cdff_dev.path import load_cdffpath
from nose.tools import assert_true, assert_equal, assert_raises_regex, \
    assert_true


def test_generate_files():
    with open("test/test_data/TiltScan_dfpc_desc.yaml", "r") as f:
        node = yaml.load(f)
    tmp_folder = "test/test_output/dfpc_ci_TiltScan"
    with EnsureCleanup(tmp_folder) as ec:
        cdffpath = load_cdffpath()
        filenames = write_dfpc(node, tmp_folder, cdffpath=cdffpath)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        assert_true(os.path.exists(os.path.join(
            tmp_folder, "TiltScanInterface.hpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "TiltScanInterface.cpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "TiltScan.hpp")))
        assert_true(os.path.exists(os.path.join(
            tmp_folder, "TiltScan.cpp")))
