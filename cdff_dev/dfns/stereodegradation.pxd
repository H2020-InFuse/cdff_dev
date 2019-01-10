from libcpp cimport bool
from cython.operator cimport dereference as deref
cimport _stereodegradation


cdef class StereoDegradation:
    cdef _stereodegradation.StereoDegradation* thisptr
    cdef bool delete_thisptr


cdef class StereoDegradationEdres:
    cdef _stereodegradation.StereoDegradationEdres* thisptr
    cdef bool delete_thisptr
