import os
import yaml
import warnings
from cdff_dev.code_generator import write_dfpc
from cdff_dev.testing import EnsureCleanup, build_extension
from nose.tools import assert_true, assert_equal, assert_raises_regex, \
    assert_true

TEST_DATA_PATH = "test/test_data/"
OUTPUT_PATH = "test/test_output/"
DFPC_NAME = "lidarSlam"
DESC_FILE = "LidarSlam_dfpc_desc.yaml"
INTERFACE_NAME = DFPC_NAME+"Interface"
PYTHON_BINDINGS_NAME = "dfpc_ci_"+DFPC_NAME


def test_generate_files():
    with open(TEST_DATA_PATH + DESC_FILE, "r") as f:
        node = yaml.load(f)
    tmp_folder = OUTPUT_PATH + DFPC_NAME
    with EnsureCleanup(tmp_folder) as ec:
        filenames = write_dfpc(node, tmp_folder)
        ec.add_files(filenames)
        ec.add_folder(os.path.join(tmp_folder, "python"))

        assert_true(
            os.path.exists(os.path.join(tmp_folder, INTERFACE_NAME + ".hpp")))
        assert_true(
            os.path.exists(os.path.join(tmp_folder, INTERFACE_NAME + ".cpp")))
        assert_true(os.path.exists(os.path.join(tmp_folder, DFPC_NAME+".hpp")))
        assert_true(os.path.exists(os.path.join(tmp_folder, DFPC_NAME+".cpp")))
        assert_true(os.path.exists(os.path.join(tmp_folder, "python",
                                                PYTHON_BINDINGS_NAME+".pxd")))
        assert_true(os.path.exists(os.path.join(tmp_folder, "python",
                                                PYTHON_BINDINGS_NAME+".pyx")))
        assert_true(os.path.exists(os.path.join(tmp_folder, "python",
                                                "_"+PYTHON_BINDINGS_NAME+".pxd")))
