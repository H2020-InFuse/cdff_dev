from libcpp cimport bool
from cython.operator cimport dereference as deref
cimport _disparitytopointcloud


cdef class DisparityToPointCloud:
    cdef _disparitytopointcloud.DisparityToPointCloud* thisptr
    cdef bool delete_thisptr


cdef class DisparityToPointCloudEdres:
    cdef _disparitytopointcloud.DisparityToPointCloudEdres* thisptr
    cdef bool delete_thisptr
