from libcpp cimport bool
from libcpp.string cimport string
cimport _cdff_types


cdef extern from "Reconstruction3DInterface.hpp" namespace "CDFF::DFPC":
    cdef cppclass Reconstruction3DInterface:
        Reconstruction3DInterface()
        void run() except +
        void setConfigurationFile(string) except +
        void setup() except +

        void leftImageInput(_cdff_types.asn1SccFrame data) except +
        void rightImageInput(_cdff_types.asn1SccFrame data) except +

        _cdff_types.asn1SccPointcloud& pointCloudOutput() except +
        _cdff_types.asn1SccPose& poseOutput() except +
        bool& successOutput() except +


cdef extern from "AdjustmentFromStereo.hpp" namespace "CDFF::DFPC::Reconstruction3D":
    cdef cppclass AdjustmentFromStereo(Reconstruction3DInterface):
        pass


cdef extern from "DenseRegistrationFromStereo.hpp" namespace "CDFF::DFPC::Reconstruction3D":
    cdef cppclass DenseRegistrationFromStereo(Reconstruction3DInterface):
        pass


cdef extern from "EstimationFromStereo.hpp" namespace "CDFF::DFPC::Reconstruction3D":
    cdef cppclass EstimationFromStereo(Reconstruction3DInterface):
        pass


cdef extern from "ReconstructionFromMotion.hpp" namespace "CDFF::DFPC::Reconstruction3D":
    cdef cppclass ReconstructionFromMotion(Reconstruction3DInterface):
        pass


cdef extern from "ReconstructionFromStereo.hpp" namespace "CDFF::DFPC::Reconstruction3D":
    cdef cppclass ReconstructionFromStereo(Reconstruction3DInterface):
        pass


cdef extern from "RegistrationFromStereo.hpp" namespace "CDFF::DFPC::Reconstruction3D":
    cdef cppclass RegistrationFromStereo(Reconstruction3DInterface):
        pass


cdef extern from "SparseRegistrationFromStereo.hpp" namespace "CDFF::DFPC::Reconstruction3D":
    cdef cppclass SparseRegistrationFromStereo(Reconstruction3DInterface):
        pass
