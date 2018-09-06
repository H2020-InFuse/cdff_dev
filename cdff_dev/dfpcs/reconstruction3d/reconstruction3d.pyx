# distutils: language=c++
cimport _reconstruction3d
cimport reconstruction3d
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string
cimport cdff_types
cimport _cdff_types


"""
cdef class Reconstruction3D:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _reconstruction3d.Reconstruction3D()
        self.delete_thisptr = True

    def run(self):
        self.thisptr.run()

    def set_configuration_file(self, str configuration_file_path):
        cdef string path = configuration_file_path.encode()
        self.thisptr.setConfigurationFile(path)

    def setup(self):
        self.thisptr.setup()

    def leftImageInput(self, cdff_types.Frame data):
        cdef _cdff_types.asn1SccFrame * cpp_data = data.thisptr
        self.thisptr.leftImageInput(deref(cpp_data))

    def rightImageInput(self, cdff_types.Frame data):
        cdef _cdff_types.asn1SccFrame * cpp_data = data.thisptr
        self.thisptr.rightImageInput(deref(cpp_data))

    def pointCloudOutput(self):
        cdef cdff_types.Pointcloud out = cdff_types.Pointcloud()
        out.thisptr[0] = self.thisptr.pointCloudOutput()
        return out

    def poseOutput(self):
        cdef cdff_types.Pose out = cdff_types.Pose()
        out.thisptr[0] = self.thisptr.poseOutput()
        return out

    def successOutput(self):
        cdef bool out = self.thisptr.successOutput()
        return out
"""
