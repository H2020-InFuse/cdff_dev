import numpy as np
from cdff_dev.extensions.gps import cdff_gps
from nose.tools import assert_equal
from numpy.testing import assert_array_equal


def test_gps():
    utm_converter = cdff_gps.UTMConverter()


def tst_gps():
    converter = cdff_gps.UTMConverter()

    utm_zone = 1
    converter.setUTMZone(utm_zone)
    converter.getUTMZone()
    assert_equal(converter.getUTMZone(), utm_zone)

    utm_north = True
    converter.setUTMNorth(utm_north)
    converter.getUTMNorth()
    assert_equal(converter.getUTMNorth(), utm_north)

    origin = cdff_types.Vector3d()
    v = np.array([0.0,1.0,2.0])
    origin.fromarray(v)
    converter.setNWUOrigin(origin)
    assert_array_equal(converter.getNWUOrigin(), origin)

    rbs_out = cdff_types.RigidBodyState()
    time = cdff_types.Time()
    time.microseconds = 1000
    lon = -110.78
    lat = 38.41
    alt = 1353.67
    rbs_out = converter.convertToUTM(time, lon, lat, alt)
    print('rbs_out.pos: ', rbs_out.pos)

    nwu_out = converter.convertToNWU(time, lon, lat, alt)
    print('nwu_out.pos: ', nwu_out.pos)


if __name__ == '__main__':
    test_gps()
