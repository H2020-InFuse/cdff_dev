from libcpp cimport bool
from cython.operator cimport dereference as deref
cimport _reconstruction3d


cdef class AdjustmentFromStereo:
    cdef _reconstruction3d.AdjustmentFromStereo* thisptr
    cdef bool delete_thisptr


cdef class DenseRegistrationFromStereo:
    cdef _reconstruction3d.DenseRegistrationFromStereo* thisptr
    cdef bool delete_thisptr


cdef class EstimationFromStereo:
    cdef _reconstruction3d.EstimationFromStereo* thisptr
    cdef bool delete_thisptr


cdef class ReconstructionFromMotion:
    cdef _reconstruction3d.ReconstructionFromMotion* thisptr
    cdef bool delete_thisptr


cdef class ReconstructionFromStereo:
    cdef _reconstruction3d.ReconstructionFromStereo* thisptr
    cdef bool delete_thisptr


cdef class RegistrationFromStereo:
    cdef _reconstruction3d.RegistrationFromStereo* thisptr
    cdef bool delete_thisptr


cdef class SparseRegistrationFromStereo:
    cdef _reconstruction3d.SparseRegistrationFromStereo* thisptr
    cdef bool delete_thisptr
