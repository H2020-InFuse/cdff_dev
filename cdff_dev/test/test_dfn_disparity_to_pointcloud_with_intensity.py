from cdff_dev.dfns.disparitytopointcloudwithintensity import DisparityToPointCloudWithIntensity
import cdff_types
from nose.tools import assert_equal


def test_disparitytopointcloud():
    dfn = DisparityToPointCloudWithIntensity()
    dfn.configure()

    disparity = cdff_types.Frame()
    disparity.data.rows = 1024
    disparity.data.cols = 768
    disparity.data.channels = 1
    disparity.array_reference()[:, :, :] = 0.0

    intensity = cdff_types.Frame()
    intensity.data.rows = 1024
    intensity.data.cols = 768
    intensity.data.channels = 1
    intensity.array_reference()[:, :, :] = 0.0

    dfn.dispImageInput(disparity)
    dfn.intensityImageInput(intensity)
    dfn.process()
    pointcloud = dfn.pointCloudOutput()
    assert_equal(pointcloud.data.points.size(), 0)