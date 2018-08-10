# distutils: language=c++
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "ogr_spatialref.h":
    cdef cppclass OGRCoordinateTransformation:
        OGRCoordinateTransformation()
        int Transform(int nCount, double* x, double* y, double* z)
    cdef cppclass OGRSpatialReference:
        OGRSpatialReference()
        int SetWellKnownGeogCS(const char* pszName)
        int SetUTM(int nZone, int bNorth)
    OGRCoordinateTransformation* OGRCreateCoordinateTransformation(
        OGRSpatialReference* poSource, OGRSpatialReference* poTarget)


def convert_to_utm(double latitude, double longitude, double altitude,
                   int utm_zone=32, bool utm_north=True):
    """Convert GPS coordinates to Universal Transverse Mercator coordinates.

    Parameters
    ----------
    latitude : double
        Latitude in degrees

    longitude : double
        Longitude in degrees

    altitude : double
        Altitude

    utm_zone : int, optional (default: 32 (central Europe))
        UTM zone, see
        https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system

    utm_north : bool, optional (default: True)
        Northern hemisphere

    Returns
    -------
    easting : double
        x

    northing : double
        y

    alt : double
        z
    """
    cdef OGRSpatialReference source_srs
    cdef OGRSpatialReference target_srs
    source_srs.SetWellKnownGeogCS("WGS84")
    target_srs.SetWellKnownGeogCS("WGS84")
    target_srs.SetUTM(utm_zone, utm_north)

    cdef OGRCoordinateTransformation* transform = \
        OGRCreateCoordinateTransformation(&source_srs, &target_srs)

    if transform == NULL:
        raise Exception("Failed to initialize coordinate transform")

    cdef double northing = latitude
    cdef double easting  = longitude
    cdef double alt = altitude

    cdef int success = transform.Transform(1, &easting, &northing, &alt)

    del transform

    if not success:
        raise Exception("Failed to transform from GPS to UTM coordinates")

    return easting, northing, alt


def convert_to_nwu(double easting, double northing, double altitude,
                   double origin_northing, double origin_westing,
                   double origin_up):
    """Convert UTM coordinates to NWU coordinates.

    Parameters
    ----------
    easting : double
        x

    northing : double
        y

    altitude : double
        z

    origin_northing : double
        x-coordinate of the base coordinate system in NWU convention

    origin_westing : double
        y-coordinate of the base coordinate system in NWU convention

    origin_up : double
        z-coordinate of the base coordinate system in NWU convention

    Returns
    -------
    northing : double
        x-coordinate in base coordinate system in NWU convention

    westing : double
        y-coordinate in base coordinate system in NWU convention

    up : double
        z-coordinate in base coordinate system in NWU convention
    """
    return (northing - origin_northing, 1000000 - easting - origin_westing,
            altitude - origin_up)
