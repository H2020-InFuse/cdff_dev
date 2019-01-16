from cdff_dev.dfns.imagedegradation import ImageDegradation
import cdff_types
from nose.tools import assert_equal


def test_imagedegradation():
    dfn = ImageDegradation()
    dfn.configure()

    original_image = cdff_types.Frame()
    original_image.data.rows = 1024
    original_image.data.cols = 768
    original_image.data.channels = 3
    original_image.array_reference()[:, :, :] = 0.0
    dfn.originalImageInput(original_image)
    dfn.process()
    degraded_image = dfn.degradedImageOutput()
    assert_equal(degraded_image.data.rows, 1024 / 2)
    assert_equal(degraded_image.data.cols, 768 / 2)
    assert_equal(degraded_image.data.channels, 3)