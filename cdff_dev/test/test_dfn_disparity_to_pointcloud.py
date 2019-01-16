from cdff_dev.dfns.disparitytopointcloud import DisparityToPointCloud
import cdff_types
from nose.tools import assert_equal


def test_disparitytopointcloud():
    dfn = DisparityToPointCloud()
    dfn.configure()

    disparity = cdff_types.Frame()
    disparity.data.rows = 1024
    disparity.data.cols = 768
    disparity.array_reference()[:, :, :] = 0.0
    dfn.dispImageInput(disparity)
    dfn.process()
    pointcloud = dfn.pointCloudOutput()
    assert_equal(pointcloud.data.points.size(), 0)