from libcpp cimport bool
from cython.operator cimport dereference as deref
cimport _disparityimage


cdef class DisparityImage:
    cdef _disparityimage.DisparityImage* thisptr
    cdef bool delete_thisptr


cdef class DisparityImageEdres:
    cdef _disparityimage.DisparityImageEdres* thisptr
    cdef bool delete_thisptr
