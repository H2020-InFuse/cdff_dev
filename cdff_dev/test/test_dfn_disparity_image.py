from cdff_dev.dfns.disparityimage import DisparityImage
import cdff_types
from nose.tools import assert_equal


def test_disparityimage():
    dfn = DisparityImage()
    dfn.configure()

    frame_pair = cdff_types.FramePair()
    frame_pair.left.data.rows = 1024
    frame_pair.left.data.cols = 768
    frame_pair.left.data.channels = 3
    frame_pair.left.array_reference()[:, :, :] = 0.0
    frame_pair.right.data.rows = 1024
    frame_pair.right.data.cols = 768
    frame_pair.right.data.channels = 3
    frame_pair.right.array_reference()[:, :, :] = 0.0
    dfn.framePairInput(frame_pair)
    dfn.process()
    disparity = dfn.rawDisparityOutput()
    assert_equal(disparity.data.rows, 1024)
    assert_equal(disparity.data.cols, 768)
    assert_equal(disparity.data.channels, 1)