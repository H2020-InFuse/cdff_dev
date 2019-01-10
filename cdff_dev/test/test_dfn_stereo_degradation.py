from cdff_dev.dfns.stereodegradation import StereoDegradation
import cdff_types
from nose.tools import assert_equal


def test_stereodegradation():
    dfn = StereoDegradation()
    dfn.configure()

    original_image_pair = cdff_types.FramePair()
    original_image_pair.left.data.rows = 1024
    original_image_pair.left.data.cols = 768
    original_image_pair.left.data.channels = 3
    original_image_pair.left.array_reference()[:, :, :] = 0.0
    original_image_pair.right.data.rows = 1024
    original_image_pair.right.data.cols = 768
    original_image_pair.right.data.channels = 3
    original_image_pair.right.array_reference()[:, :, :] = 0.0
    dfn.originalImagePairInput(original_image_pair)
    dfn.process()
    degraded_image_pair = dfn.degradedImagePairOutput()
    assert_equal(degraded_image_pair.left.data.rows, 1024 / 2)
    assert_equal(degraded_image_pair.left.data.cols, 768 / 2)
    assert_equal(degraded_image_pair.left.data.channels, 3)
    assert_equal(degraded_image_pair.right.data.rows, 1024 / 2)
    assert_equal(degraded_image_pair.right.data.cols, 768 / 2)
    assert_equal(degraded_image_pair.right.data.channels, 3)