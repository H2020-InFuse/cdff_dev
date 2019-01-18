from libcpp cimport bool
from cython.operator cimport dereference as deref
cimport _disparitytopointcloudwithintensity


cdef class DisparityToPointCloudWithIntensity:
    cdef _disparitytopointcloudwithintensity.DisparityToPointCloudWithIntensity* thisptr
    cdef bool delete_thisptr


cdef class DisparityToPointCloudWithIntensityEdres:
    cdef _disparitytopointcloudwithintensity.DisparityToPointCloudWithIntensityEdres* thisptr
    cdef bool delete_thisptr
