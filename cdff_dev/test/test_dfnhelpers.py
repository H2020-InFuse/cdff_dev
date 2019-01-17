import numpy as np
from cdff_dev import dfnhelpers, dataflowcontrol
import cdff_types
from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal


def test_lambda_dfn():
    dfn = dfnhelpers.LambdaDFN(lambda x: x ** 2, "x", "y")
    assert_true(dataflowcontrol.isdfn(dfn))
    dfn.set_configuration_file("")
    dfn.configure()
    dfn.xInput(5)
    dfn.process()
    y = dfn.yOutput()
    assert_equal(y, 25)


def test_merge_without_config():
    merger = dfnhelpers.MergeFramePairDFN()
    merger.configure()

    random_state = np.random.RandomState(0)

    left = cdff_types.Frame()
    left.data.channels = 1
    left.data.cols = 2
    left.data.rows = 3
    left_image = left.array_reference()
    left_image[:, :, :] = random_state.randint(
        0, 256, (3, 2, 1), dtype=np.uint8)

    right = cdff_types.Frame()
    right.data.channels = 1
    right.data.cols = 2
    right.data.rows = 3
    right_image = right.array_reference()
    right_image[:, :, :] = random_state.randint(
        0, 256, (3, 2, 1), dtype=np.uint8)

    merger.leftImageInput(left)
    merger.rightImageInput(right)
    merger.process()
    pair = merger.pairOutput()

    assert_equal(pair.left.data.channels, left.data.channels)
    assert_equal(pair.left.data.cols, left.data.cols)
    assert_equal(pair.left.data.rows, left.data.rows)
    assert_array_equal(pair.left.array_reference(), left_image)

    assert_equal(pair.right.data.channels, right.data.channels)
    assert_equal(pair.right.data.cols, right.data.cols)
    assert_equal(pair.right.data.rows, right.data.rows)
    assert_array_equal(pair.right.array_reference(), right_image)


def test_merge_with_config():
    merger = dfnhelpers.MergeFramePairDFN(
        left_camera_info_stream="/hcru1/pt_stereo_rect/left/camera_info",
        right_camera_info_stream="/hcru1/pt_stereo_rect/right/camera_info",
        left_is_main_camera=True
    )
    merger.set_configuration_file("test/test_data/camera_config.msg")
    merger.configure()

    left = cdff_types.Frame()
    right = cdff_types.Frame()

    merger.leftImageInput(left)
    merger.rightImageInput(right)
    merger.process()
    pair = merger.pairOutput()

    dist_coeffs = pair.left.intrinsic.dist_coeffs.toarray()
    assert_array_equal(dist_coeffs, np.array([-0.188116, 0.227419, 0, 0, 0]))
    camera_matrix = pair.left.intrinsic.camera_matrix
    assert_array_equal(
        camera_matrix,
        np.array([[867.356, 0, 516], [0, 867.356, 386], [0, 0, 1]]))
    assert_equal(pair.left.intrinsic.camera_model, "cam_PINHOLE")

    dist_coeffs = pair.right.intrinsic.dist_coeffs.toarray()
    assert_array_equal(dist_coeffs, np.array([-0.183758, 0.221971, 0, 0, 0]))
    camera_matrix = pair.right.intrinsic.camera_matrix
    assert_array_equal(
        camera_matrix,
        np.array([[867.356, 0, 516], [0, 867.356, 386], [0, 0, 1]]))
    assert_equal(pair.right.intrinsic.camera_model, "cam_PINHOLE")
