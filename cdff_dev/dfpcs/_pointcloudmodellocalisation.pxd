from libcpp cimport bool
from libcpp.string cimport string
cimport _cdff_types


cdef extern from "PointCloudModelLocalisationInterface.hpp" namespace "CDFF::DFPC":
    cdef cppclass PointCloudModelLocalisationInterface:
        PointCloudModelLocalisationInterface()
        void run() except +
        void setConfigurationFile(string) except +
        void setup() except +

        void sceneInput(_cdff_types.asn1SccPointcloud data) except +
        void modelInput(_cdff_types.asn1SccPointcloud data) except +
        void computeModelFeaturesInput(bool data) except +

        _cdff_types.asn1SccPose& poseOutput() except +
        bool& successOutput() except +



cdef extern from "FeaturesMatching3D.hpp" namespace "CDFF::DFPC::PointCloudModelLocalisation":
    cdef cppclass FeaturesMatching3D(PointCloudModelLocalisationInterface):
        pass
