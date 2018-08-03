from libcpp cimport bool
from cython.operator cimport dereference as deref
cimport _cdff_gps


cdef class UTMConverter:
    cdef _cdff_gps.UTMConverter* thisptr
    cdef bool delete_thisptr
