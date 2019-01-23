from libcpp cimport bool
from libcpp.string cimport string
cimport _cdff_types


cdef extern from "DisparityFilteringInterface.hpp" namespace "CDFF::DFN":
    cdef cppclass DisparityFilteringInterface:
        DisparityFilteringInterface()
        void process() except +
        void setConfigurationFile(string) except +
        void configure() except +

        void dispImageInput(_cdff_types.asn1SccFrame& data) except +

        _cdff_types.asn1SccFrame& filteredDispImageOutput() except +


cdef extern from "DisparityFiltering.hpp" namespace "CDFF::DFN::DisparityFiltering":
    cdef cppclass DisparityFiltering(DisparityFilteringInterface):
        pass
