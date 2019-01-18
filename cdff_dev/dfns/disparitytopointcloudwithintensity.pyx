# distutils: language=c++
cimport _disparitytopointcloudwithintensity
cimport disparitytopointcloudwithintensity
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string
cimport cdff_types
cimport _cdff_types


cdef class DisparityToPointCloudWithIntensity:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _disparitytopointcloudwithintensity.DisparityToPointCloudWithIntensity()
        self.delete_thisptr = True

    def process(self):
        self.thisptr.process()

    def set_configuration_file(self, str configuration_file_path):
        cdef string path = configuration_file_path.encode()
        self.thisptr.setConfigurationFile(path)

    def configure(self):
        self.thisptr.configure()

    def dispImageInput(self, cdff_types.Frame data):
        cdef _cdff_types.asn1SccFrame * cpp_data = data.thisptr
        self.thisptr.dispImageInput(deref(cpp_data))

    def intensityImageInput(self, cdff_types.Frame data):
        cdef _cdff_types.asn1SccFrame * cpp_data = data.thisptr
        self.thisptr.intensityImageInput(deref(cpp_data))

    def pointCloudOutput(self):
        cdef cdff_types.Pointcloud out = cdff_types.Pointcloud()
        out.thisptr[0] = self.thisptr.pointCloudOutput()
        return out


"""
cdef class DisparityToPointCloudWithIntensityEdres:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.delete_thisptr and self.thisptr != NULL:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _disparitytopointcloudwithintensity.DisparityToPointCloudWithIntensityEdres()
        self.delete_thisptr = True

    def process(self):
        self.thisptr.process()

    def set_configuration_file(self, str configuration_file_path):
        cdef string path = configuration_file_path.encode()
        self.thisptr.setConfigurationFile(path)

    def configure(self):
        self.thisptr.configure()

    def dispImageInput(self, cdff_types.Frame data):
        cdef _cdff_types.asn1SccFrame * cpp_data = data.thisptr
        self.thisptr.dispImageInput(deref(cpp_data))

    def intensityImageInput(self, cdff_types.Frame data):
        cdef _cdff_types.asn1SccFrame * cpp_data = data.thisptr
        self.thisptr.intensityImageInput(deref(cpp_data))

    def pointCloudOutput(self):
        cdef cdff_types.Pointcloud out = cdff_types.Pointcloud()
        out.thisptr[0] = self.thisptr.pointCloudOutput()
        return out
"""
