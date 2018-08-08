from cdff_dev.extensions.gps import conversion
from nose.tools import assert_equal, assert_almost_equal


def test_gps_to_utm_utah():
    # source: https://www.latlong.net/place/utah-usa-5877.html
    easting, northing, altitude = conversion.convert_to_utm(
        39.419220, -111.950684, 1580.0, utm_zone=12, utm_north=True)
    assert_almost_equal(easting, 418165.85, places=2)
    assert_almost_equal(northing, 4363730.68, places=2)
    assert_equal(altitude, 1580.0)


def test_gps_to_utm_bremen():
    # source: https://www.latlong.net/place/bremen-germany-11253.html
    easting, northing, altitude = conversion.convert_to_utm(
        53.073635, 8.806422, 12.0, utm_zone=32, utm_north=True)
    assert_almost_equal(easting, 487031.03, places=2)
    assert_almost_equal(northing, 5880479.38, places=2)
    assert_equal(altitude, 12.0)


def test_gps_to_utm_sydney():
    # source: https://www.latlong.net/place/sydney-nsw-australia-700.html
    easting, northing, altitude = conversion.convert_to_utm(
        -33.865143, 151.209900, 19.0, utm_zone=56, utm_north=False)
    assert_almost_equal(easting, 334417.07, places=2)
    assert_almost_equal(northing, 6251354.86, places=2)
    assert_equal(altitude, 19.0)


def test_utm_to_nwu_without_origin():
    easting, northing, altitude = 418165.85, 4363730.68, 1580.0
    northing, westing, up = conversion.convert_to_nwu(
        easting, northing, altitude, 0.0, 0.0, 0.0)
    assert_almost_equal(northing, 4363730.68)
    assert_almost_equal(westing, 581834.15)
    assert_equal(up, 1580.0)


def test_utm_to_nwu_with_origin():
    easting, northing, altitude = 487031.03, 5880479.38, 12.0
    northing, westing, up = conversion.convert_to_nwu(
        easting, northing, altitude, 5880479.0, 512968.0, 12.0)
    assert_almost_equal(northing, 0.38)
    assert_almost_equal(westing, 0.97)
    assert_equal(up, 0.0)


if __name__ == '__main__':
    test_gps_to_utm_utah()
    test_gps_to_utm_bremen()
    test_gps_to_utm_sydney()
    test_utm_to_nwu_without_origin()
    test_utm_to_nwu_with_origin()
