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


cdef class Vector3d:
    cdef _cdff_types.Vector3d* thisptr
    cdef bool delete_thisptr


cdef class Vector4d:
    cdef _cdff_types.Vector4d* thisptr
    cdef bool delete_thisptr


cdef class Vector6d:
    cdef _cdff_types.Vector6d* thisptr
    cdef bool delete_thisptr


cdef class VectorXd:
    cdef _cdff_types.VectorXd* thisptr
    cdef bool delete_thisptr


cdef class Matrix2d:
    cdef _cdff_types.Matrix2d* thisptr
    cdef bool delete_thisptr


cdef class Matrix3d:
    cdef _cdff_types.Matrix3d* thisptr
    cdef bool delete_thisptr


cdef class Quaterniond:
    cdef _cdff_types.Quaterniond* thisptr
    cdef bool delete_thisptr


cdef class AngleAxisd:
    cdef _cdff_types.AngleAxisd* thisptr
    cdef bool delete_thisptr


cdef class Transform3d:
    cdef _cdff_types.Transform3d* thisptr
    cdef bool delete_thisptr
