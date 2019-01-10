from libcpp cimport bool
from libcpp.string cimport string
cimport _cdff_types


cdef extern from "StereoDegradationInterface.hpp" namespace "CDFF::DFN":
    cdef cppclass StereoDegradationInterface:
        StereoDegradationInterface()
        void process() except +
        void setConfigurationFile(string) except +
        void configure() except +

        void originalImagePairInput(_cdff_types.asn1SccFramePair& data) except +

        _cdff_types.asn1SccFramePair& degradedImagePairOutput() except +


cdef extern from "StereoDegradation.hpp" namespace "CDFF::DFN::StereoDegradation":
    cdef cppclass StereoDegradation(StereoDegradationInterface):
        pass


cdef extern from "StereoDegradationEdres.hpp" namespace "CDFF::DFN::StereoDegradation":
    cdef cppclass StereoDegradationEdres(StereoDegradationInterface):
        pass
