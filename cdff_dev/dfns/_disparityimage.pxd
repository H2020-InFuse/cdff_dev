from libcpp cimport bool
from libcpp.string cimport string
cimport _cdff_types


cdef extern from "DisparityImageInterface.hpp" namespace "CDFF::DFN":
    cdef cppclass DisparityImageInterface:
        DisparityImageInterface()
        void process() except +
        void setConfigurationFile(string) except +
        void configure() except +

        void framePairInput(_cdff_types.asn1SccFramePair& data) except +

        _cdff_types.asn1SccFrame& disparityOutput() except +


cdef extern from "DisparityImage.hpp" namespace "CDFF::DFN::DisparityImage":
    cdef cppclass DisparityImage(DisparityImageInterface):
        pass


cdef extern from "DisparityImageEdres.hpp" namespace "CDFF::DFN::DisparityImage":
    cdef cppclass DisparityImageEdres(DisparityImageInterface):
        pass
