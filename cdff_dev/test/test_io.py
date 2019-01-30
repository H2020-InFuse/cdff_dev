from cdff_dev import io
from nose.tools import assert_equal, assert_almost_equal


def test_geotiff_slice():
    gtm = io.GeoTiffMap("test/test_data/maps/res1.tif", verbose=0)
    origin, m = gtm.slice((250, 500), (600, 800))
    assert_equal(m.data.rows, 200)
    assert_equal(m.data.cols, 250)
    assert_equal(m.data.channels, 1)
    assert_equal(m.data.depth, "depth_32F")
    mean = m.data.array_reference()[
        m.data.array_reference() != gtm.undefined].mean()
    assert_almost_equal(mean, 47.84948, places=5)


def test_geotiff_downsampling():
    gtm = io.GeoTiffMap("test/test_data/maps/res1.tif", verbose=0)
    m = gtm.downsample(2)
    assert_equal(m.data.rows, 707)
    assert_equal(m.data.cols, 328)
    assert_equal(m.data.channels, 1)
    assert_equal(m.data.depth, "depth_32F")
    mean = m.data.array_reference()[
        m.data.array_reference() != gtm.undefined].mean()
    assert_almost_equal(mean, 48.726055, places=6)
