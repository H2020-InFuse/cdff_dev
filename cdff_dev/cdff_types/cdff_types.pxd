# distutils: language = c++
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
cimport _cdff_types


cdef class Time:
    cdef _cdff_types.Time* thisptr
    cdef bool delete_thisptr


cdef class Vector2d:
    cdef _cdff_types.Vector2d* thisptr
    cdef bool delete_thisptr
