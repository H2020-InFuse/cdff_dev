from libcpp cimport bool
from libcpp.string cimport string
cimport _cdff_types


cdef extern from "DisparityToPointCloudWithIntensityInterface.hpp" namespace "CDFF::DFN":
    cdef cppclass DisparityToPointCloudWithIntensityInterface:
        DisparityToPointCloudWithIntensityInterface()
        void process() except +
        void setConfigurationFile(string) except +
        void configure() except +

        void dispImageInput(_cdff_types.asn1SccFrame& data) except +
        void intensityImageInput(_cdff_types.asn1SccFrame& data) except +

        _cdff_types.asn1SccPointcloud& pointCloudOutput() except +


cdef extern from "DisparityToPointCloudWithIntensity.hpp" namespace "CDFF::DFN::DisparityToPointCloudWithIntensity":
    cdef cppclass DisparityToPointCloudWithIntensity(DisparityToPointCloudWithIntensityInterface):
        pass


cdef extern from "DisparityToPointCloudWithIntensityEdres.hpp" namespace "CDFF::DFN::DisparityToPointCloudWithIntensity":
    cdef cppclass DisparityToPointCloudWithIntensityEdres(DisparityToPointCloudWithIntensityInterface):
        pass
