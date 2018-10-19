# distutils: language=c++
cimport _pointcloudmodellocalisation
cimport pointcloudmodellocalisation
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string
cimport cdff_types
cimport _cdff_types


cdef class FeaturesMatching3D:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _pointcloudmodellocalisation.FeaturesMatching3D()
        self.delete_thisptr = True

    def run(self):
        self.thisptr.run()

    def set_configuration_file(self, str configuration_file_path):
        cdef string path = configuration_file_path.encode()
        self.thisptr.setConfigurationFile(path)

    def setup(self):
        self.thisptr.setup()

    def sceneInput(self, cdff_types.Pointcloud data):
        cdef _cdff_types.asn1SccPointcloud * cpp_data = data.thisptr
        self.thisptr.sceneInput(deref(cpp_data))

    def modelInput(self, cdff_types.Pointcloud data):
        cdef _cdff_types.asn1SccPointcloud * cpp_data = data.thisptr
        self.thisptr.modelInput(deref(cpp_data))

    def computeModelFeaturesInput(self, bool data):
        cdef bool cpp_data = data
        self.thisptr.computeModelFeaturesInput(cpp_data)

    def poseOutput(self):
        cdef cdff_types.Pose out = cdff_types.Pose()
        out.thisptr[0] = self.thisptr.poseOutput()
        return out

    def successOutput(self):
        cdef bool out = self.thisptr.successOutput()
        return out

