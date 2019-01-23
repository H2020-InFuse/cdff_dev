from libcpp cimport bool
from cython.operator cimport dereference as deref
cimport _disparityfiltering


cdef class DisparityFiltering:
    cdef _disparityfiltering.DisparityFiltering* thisptr
    cdef bool delete_thisptr
