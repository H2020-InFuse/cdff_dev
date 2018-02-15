# distutils: language = c++
cimport cdff_types
cimport _cdff_types
from cython.operator cimport dereference as deref
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
cimport numpy as np
import numpy as np


np.import_array()  # must be here because we use the NumPy C API


cdef class Time:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.Time()
        self.delete_thisptr = True

    def __str__(self):
        return str("{type: Time, microseconds: %d, usec_per_sec: %d}" % (self.thisptr.microseconds, self.thisptr.usecPerSec))

    def assign(self, Time other):
        self.thisptr.assign(deref(other.thisptr))

    def _get_microseconds(self):
        return self.thisptr.microseconds

    def _set_microseconds(self, int64_t microseconds):
        self.thisptr.microseconds = microseconds

    microseconds = property(_get_microseconds, _set_microseconds)

    def _get_usec_per_sec(self):
        return self.thisptr.usecPerSec

    def _set_usec_per_sec(self, int64_t usecPerSec):
        self.thisptr.usecPerSec = usecPerSec

    usec_per_sec = property(_get_usec_per_sec, _set_usec_per_sec)


cdef class Vector2d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.Vector2d()
        self.thisptr.nCount = 2
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("{type: Vector2d, data=[%.2f, %.2f]}"
                   % (self.thisptr.arr[0], self.thisptr.arr[1]))

    def __array__(self, dtype=None):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 2
        return np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_DOUBLE, <void*> self.thisptr.arr)

    def __getitem__(self, int i):
        if i < 0 or i > 1:
            raise KeyError("index must be 0 or 1 but was %d" % i)
        return self.thisptr.arr[i]

    def __setitem__(self, int i, double v):
        if i < 0 or i > 1:
            raise KeyError("index must be 0 or 1 but was %d" % i)
        self.thisptr.arr[i] = v

    def assign(self, Vector2d other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=1] array = np.empty(2)
        cdef int i
        for i in range(2):
            array[i] = self.thisptr.arr[i]
        return array

    def fromarray(self, np.ndarray[double, ndim=1] array):
        cdef int i
        for i in range(2):
            self.thisptr.arr[i] = array[i]


cdef class Vector3d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.Vector3d()
        self.thisptr.nCount = 3
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("{type: Vector3d, data=[%.2f, %.2f, %.2f]}"
                   % (self.thisptr.arr[0], self.thisptr.arr[1],
                      self.thisptr.arr[2]))

    def __array__(self, dtype=None):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 3
        return np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_DOUBLE, <void*> self.thisptr.arr)

    def __getitem__(self, int i):
        if i < 0 or i > 2:
            raise KeyError("index must be 0, 1, or 2 but was %d" % i)
        return self.thisptr.arr[i]

    def __setitem__(self, int i, double v):
        if i < 0 or i > 2:
            raise KeyError("index must be 0, 1, or 2 but was %d" % i)
        self.thisptr.arr[i] = v

    def assign(self, Vector3d other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=1] array = np.empty(3)
        cdef int i
        for i in range(3):
            array[i] = self.thisptr.arr[i]
        return array

    def fromarray(self, np.ndarray[double, ndim=1] array):
        cdef int i
        for i in range(3):
            self.thisptr.arr[i] = array[i]


cdef class Vector4d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.Vector4d()
        self.thisptr.nCount = 4
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("{type: Vector4d, data=[%.2f, %.2f, %.2f, %.2f]}"
                   % (self.thisptr.arr[0], self.thisptr.arr[1],
                      self.thisptr.arr[2], self.thisptr.arr[3]))

    def __array__(self, dtype=None):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 4
        return np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_DOUBLE, <void*> self.thisptr.arr)

    def __getitem__(self, int i):
        if i < 0 or i > 3:
            raise KeyError("index must be 0, 1, or 2 but was %d" % i)
        return self.thisptr.arr[i]

    def __setitem__(self, int i, double v):
        if i < 0 or i > 3:
            raise KeyError("index must be 0, 1, or 2 but was %d" % i)
        self.thisptr.arr[i] = v

    def assign(self, Vector4d other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=1] array = np.empty(4)
        cdef int i
        for i in range(4):
            array[i] = self.thisptr.arr[i]
        return array

    def fromarray(self, np.ndarray[double, ndim=1] array):
        cdef int i
        for i in range(4):
            self.thisptr.arr[i] = array[i]


cdef class Vector6d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.Vector6d()
        self.thisptr.nCount = 6
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("{type: Vector6d, data=[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]}"
                   % (self.thisptr.arr[0], self.thisptr.arr[1],
                      self.thisptr.arr[2], self.thisptr.arr[3],
                      self.thisptr.arr[4], self.thisptr.arr[5]))

    def __array__(self, dtype=None):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 6
        return np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_DOUBLE, <void*> self.thisptr.arr)

    def __getitem__(self, int i):
        if i < 0 or i > 5:
            raise KeyError("index must be in [0, 5] but was %d" % i)
        return self.thisptr.arr[i]

    def __setitem__(self, int i, double v):
        if i < 0 or i > 5:
            raise KeyError("index must be in [0, 5] but was %d" % i)
        self.thisptr.arr[i] = v

    def assign(self, Vector6d other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=1] array = np.empty(6)
        cdef int i
        for i in range(6):
            array[i] = self.thisptr.arr[i]
        return array

    def fromarray(self, np.ndarray[double, ndim=1] array):
        cdef int i
        for i in range(6):
            self.thisptr.arr[i] = array[i]


cdef class VectorXd:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self, int n_count=1):
        self.thisptr = new _cdff_types.VectorXd()
        self.thisptr.nCount = n_count
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("{type: VectorXd, data=[%s]}"
                   % ", ".join(["%.2f" % self.thisptr.arr[i]
                                for i in range(self.thisptr.nCount)]))

    def __array__(self, dtype=None):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.thisptr.nCount
        return np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_DOUBLE, <void*> self.thisptr.arr)

    def __getitem__(self, int i):
        if i < 0 or i >= self.thisptr.nCount:
            raise KeyError("index out of range: %d" % i)
        return self.thisptr.arr[i]

    def __setitem__(self, int i, double v):
        if i < 0 or i >= self.thisptr.nCount:
            raise KeyError("index out of range: %d" % i)
        self.thisptr.arr[i] = v

    def assign(self, VectorXd other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=1] array = np.empty(self.thisptr.nCount)
        cdef int i
        for i in range(self.thisptr.nCount):
            array[i] = self.thisptr.arr[i]
        return array

    def fromarray(self, np.ndarray[double, ndim=1] array):
        if len(array) > 100:
            raise ValueError("VectorXd supports a maximum length of 100!")
        self.thisptr.nCount = len(array)
        cdef int i
        for i in range(self.thisptr.nCount):
            self.thisptr.arr[i] = array[i]


cdef class Matrix2d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.Matrix2d()
        self.thisptr.nCount = 2
        cdef int i
        for i in range(self.thisptr.nCount):
            self.thisptr.arr[i].nCount = self.thisptr.nCount
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("{type: Matrix2d, data=[?]}") # TODO print content

    def __array__(self, dtype=None):
        return self.toarray().astype(dtype)

    def __getitem__(self, tuple indices):
        cdef int i, j
        i, j = indices
        if i < 0 or i >= self.thisptr.nCount:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= self.thisptr.nCount:
            raise KeyError("index out of range %d" % j)
        return self.thisptr.arr[i].arr[j]

    def __setitem__(self, tuple indices, double v):
        cdef int i, j
        i, j = indices
        if i < 0 or i >= self.thisptr.nCount:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= self.thisptr.nCount:
            raise KeyError("index out of range %d" % j)
        self.thisptr.arr[i].arr[j] = v

    def assign(self, Matrix2d other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=2] array = np.empty((2, 2))
        cdef int i, j
        for i in range(self.thisptr.nCount):
            for j in range(self.thisptr.nCount):
                array[i, j] = self.thisptr.arr[i].arr[j]
        return array

    def fromarray(self, np.ndarray[double, ndim=2] array):
        cdef int i, j
        for i in range(self.thisptr.nCount):
            for j in range(self.thisptr.nCount):
                self.thisptr.arr[i].arr[j] = array[i, j]