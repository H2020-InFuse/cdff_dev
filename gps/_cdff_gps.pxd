from libcpp cimport bool
cimport _cdff_types


cdef extern from "UTMConverter.hpp" namespace "gps":
    cdef cppclass UTMConverter:
        UTMConverter()
        void setUTMZone(int zone)
        void setUTMNorth(bool north)
        int getUTMZone()
        bool getUTMNorth()
        void setNWUOrigin(_cdff_types.asn1SccVector3d origin)
        _cdff_types.asn1SccVector3d getNWUOrigin()
        _cdff_types.asn1SccRigidBodyState convertToUTM(_cdff_types.asn1SccTime time, double longitude, double latitude, double altitude)
        _cdff_types.asn1SccRigidBodyState convertToNWU(_cdff_types.asn1SccTime time, double longitude, double latitude, double altitude)
