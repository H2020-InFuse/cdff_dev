# distutils: language=c++
cimport _cdff_gps
cimport cdff_gps
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string
cimport cdff_types
cimport _cdff_types

cdef class UTMConverter:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_gps.UTMConverter()
        self.delete_thisptr = True

    #setUTMZone(int zone)
    def setUTMZone(self, int zone):
        cdef int value = zone
        self.thisptr.setUTMZone(value)

    #setUTMNorth(bool north)
    def setUTMNorth(self, bool north):
        cdef bool value = north
        self.thisptr.setUTMNorth(value)

    #getUTMZone()
    def getUTMZone(self):
        cdef int out = self.thisptr.getUTMZone()
        return out

    #getUTMNorth()
    def getUTMNorth(self):
        cdef bool out = self.thisptr.getUTMNorth()
        return out

    #setNWUOrigin(_cdff_types.asn1SccPosition origin)
    def setNWUOrigin(self, cdff_types.Vector3d origin):
        cdef _cdff_types.asn1SccVector3d * cpp_data = origin.thisptr
        self.thisptr.setNWUOrigin(deref(cpp_data))

    #getNWUOrigin()
    def getNWUOrigin(self):
        cdef cdff_types.Vector3d out = cdff_types.Vector3d()
        out.thisptr[0] = self.thisptr.getNWUOrigin()
        return out

    #convertToUTM(_cdff_types.asn1SccTime time, double longitude, double latitude, double altitude)
    def convertToUTM(self, cdff_types.Time time, double longitude,
                    double latitude, double altitude):
        cdef _cdff_types.asn1SccTime * cpp_time = time.thisptr
        cdef float lon = longitude
        cdef float lat = latitude
        cdef float alt = altitude
        cdef cdff_types.RigidBodyState out = cdff_types.RigidBodyState()
        out.thisptr[0] = self.thisptr.convertToUTM(deref(cpp_time), lon,
                                                        lat, alt)
        return out

    #convertToNWU(_cdff_types.asn1SccTime time, double longitude, double latitude, double altitude)
    def convertToNWU(self, cdff_types.Time time, double longitude,
                    double latitude, double altitude):
        cdef _cdff_types.asn1SccTime * cpp_time = time.thisptr
        cdef float lon = longitude
        cdef float lat = latitude
        cdef float alt = altitude
        cdef cdff_types.RigidBodyState out = cdff_types.RigidBodyState()
        out.thisptr[0] = self.thisptr.convertToNWU(deref(cpp_time), lon,
                                                        lat, alt)
        return out
