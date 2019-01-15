from libcpp cimport bool
from libcpp.string cimport string
cimport _cdff_types


cdef extern from "DisparityToPointCloudInterface.hpp" namespace "CDFF::DFN":
    cdef cppclass DisparityToPointCloudInterface:
        DisparityToPointCloudInterface()
        void process() except +
        void setConfigurationFile(string) except +
        void configure() except +

        void dispImageInput(_cdff_types.asn1SccFrame& data) except +

        _cdff_types.asn1SccPointcloud& pointCloudOutput() except +


cdef extern from "DisparityToPointCloud.hpp" namespace "CDFF::DFN::DisparityToPointCloud":
    cdef cppclass DisparityToPointCloud(DisparityToPointCloudInterface):
        pass


cdef extern from "DisparityToPointCloudEdres.hpp" namespace "CDFF::DFN::DisparityToPointCloud":
    cdef cppclass DisparityToPointCloudEdres(DisparityToPointCloudInterface):
        pass
