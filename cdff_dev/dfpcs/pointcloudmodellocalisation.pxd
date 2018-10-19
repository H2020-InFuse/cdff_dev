from libcpp cimport bool
from cython.operator cimport dereference as deref
cimport _pointcloudmodellocalisation


cdef class FeaturesMatching3D:
    cdef _pointcloudmodellocalisation.FeaturesMatching3D* thisptr
    cdef bool delete_thisptr
