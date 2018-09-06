from libcpp cimport bool
from libcpp.string cimport string
cimport _cdff_types


cdef extern from "Reconstruction3DInterface.hpp" namespace "CDFF::DFPC":
    cdef cppclass Reconstruction3DInterface:
        Reconstruction3DInterface()
        void run()
        void setConfigurationFile(string)
        void setup()

        void leftImageInput(_cdff_types.asn1SccFrame data)
        void rightImageInput(_cdff_types.asn1SccFrame data)

        _cdff_types.asn1SccPointcloud pointCloudOutput()
        _cdff_types.asn1SccPose poseOutput()
        bool successOutput()



#cdef extern from "Reconstruction3D.hpp" namespace "CDFF::DFPC::Reconstruction3D":
#    cdef cppclass Reconstruction3D(Reconstruction3DInterface):
#        pass
