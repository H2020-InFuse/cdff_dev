from libcpp.string cimport string
from libcpp cimport bool
cimport _cdff_envire


cdef class Time:
    cdef _cdff_envire.Time* thisptr
    cdef bool delete_thisptr


cdef class Quaterniond:
    cdef _cdff_envire.Quaterniond* thisptr
    cdef bool delete_thisptr


cdef class Vector3d:
    cdef _cdff_envire.Vector3d* thisptr
    cdef bool delete_thisptr


cdef class TransformWithCovariance:
    cdef _cdff_envire.TransformWithCovariance* thisptr
    cdef bool delete_thisptr


cdef class Transform:
    cdef _cdff_envire.Transform* thisptr
    cdef bool delete_thisptr


cdef class EnvireGraph:
    cdef _cdff_envire.EnvireGraph* thisptr
    cdef bool delete_thisptr
