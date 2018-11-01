# distutils: language = c++
cimport cdff_types
cimport _cdff_types
import cython
from cython.operator cimport dereference as deref
from libc.string cimport memcpy
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np
import warnings


np.import_array()  # must be here because we use the NumPy C API


cdef class Time:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccTime()
        self.delete_thisptr = True

    def __str__(self):
        return "{type: Time, microseconds: %d}" % self.thisptr.microseconds

    def assign(self, Time other):
        self.thisptr.assign(deref(other.thisptr))

    def _get_microseconds(self):
        return self.thisptr.microseconds

    def _set_microseconds(self, microseconds):
        self.thisptr.microseconds = <int64_t> microseconds

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
        self.thisptr = new _cdff_types.asn1SccVector2d()
        self.delete_thisptr = True

    def __len__(self):
        return 2

    def __str__(self):
        return str("{type: Vector2d, data: [%g, %g]}"
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
        self.thisptr = new _cdff_types.asn1SccVector3d()
        self.delete_thisptr = True

    def __len__(self):
        return 3

    def __str__(self):
        return str("{type: Vector3d, data: [%g, %g, %g]}"
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
        self.thisptr = new _cdff_types.asn1SccVector4d()
        self.delete_thisptr = True

    def __len__(self):
        return 4

    def __str__(self):
        return str("{type: Vector4d, data: [%g, %g, %g, %g]}"
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
        self.thisptr = new _cdff_types.asn1SccVector6d()
        self.delete_thisptr = True

    def __len__(self):
        return 6

    def __str__(self):
        return ("{type: Vector6d, data: [%g, %g, %g, %g, %g, %g]}"
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
        self.thisptr = new _cdff_types.asn1SccVectorXd()
        self.thisptr.nCount = n_count
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("{type: VectorXd, data: [%s]}"
                   % ", ".join(["%g" % self.thisptr.arr[i]
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
        if i < 0 or i >= 100:
            raise KeyError("index out of range: %d" % i)
        if i >= self.thisptr.nCount:
            self.thisptr.nCount = i + 1
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
        self.thisptr = new _cdff_types.asn1SccMatrix2d()
        self.delete_thisptr = True

    def __len__(self):
        return 2

    def __str__(self):
        return ("{type: Matrix2d, data: [[%g, %g], [%g, %g]]}"
                % tuple(self.toarray().ravel()))

    def __array__(self, dtype=None):
        return self.toarray().astype(dtype)

    def __getitem__(self, tuple indices):
        cdef int i, j
        i, j = indices
        if i < 0 or i >= 2:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 2:
            raise KeyError("index out of range %d" % j)
        return self.thisptr.arr[i].arr[j]

    def __setitem__(self, tuple indices, double v):
        cdef int i, j
        i, j = indices
        if i < 0 or i >= 2:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 2:
            raise KeyError("index out of range %d" % j)
        self.thisptr.arr[i].arr[j] = v

    @property
    def shape(self):
        return (2, 2)

    def assign(self, Matrix2d other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=2] array = np.empty((2, 2))
        cdef int i, j
        for i in range(2):
            for j in range(2):
                array[i, j] = self.thisptr.arr[i].arr[j]
        return array

    def fromarray(self, np.ndarray[double, ndim=2] array):
        cdef int i, j
        for i in range(2):
            for j in range(2):
                self.thisptr.arr[i].arr[j] = array[i, j]


cdef class Matrix3d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccMatrix3d()
        self.delete_thisptr = True

    def __len__(self):
        return 3

    def __str__(self):
        return ("{type: Matrix3d, data: [[%g, %g, %g], [%g, %g, %g], "
                "[%g, %g, %g]]}" % tuple(self.toarray().ravel()))

    def __array__(self, dtype=None):
        return self.toarray().astype(dtype)

    def __getitem__(self, tuple indices):
        cdef int i, j
        i, j = indices
        if i < 0 or i >= 3:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 3:
            raise KeyError("index out of range %d" % j)
        return self.thisptr.arr[i].arr[j]

    def __setitem__(self, tuple indices, double v):
        cdef int i, j
        i, j = indices
        if i < 0 or i >= 3:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 3:
            raise KeyError("index out of range %d" % j)
        self.thisptr.arr[i].arr[j] = v

    @property
    def shape(self):
        return (3, 3)

    def assign(self, Matrix3d other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=2] array = np.empty((3, 3))
        cdef int i, j
        for i in range(3):
            for j in range(3):
                array[i, j] = self.thisptr.arr[i].arr[j]
        return array

    def fromarray(self, np.ndarray[double, ndim=2] array):
        cdef int i, j
        for i in range(3):
            for j in range(3):
                self.thisptr.arr[i].arr[j] = array[i, j]


cdef class Matrix6d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccMatrix6d()
        self.delete_thisptr = True

    def __len__(self):
        return 6

    def __str__(self):
        return ("{type: Matrix6d, data: [[%g, %g, %g, %g, %g, %g], "
                "[%g, %g, %g, %g, %g, %g], [%g, %g, %g, %g, %g, %g], "
                "[%g, %g, %g, %g, %g, %g], [%g, %g, %g, %g, %g, %g], "
                "[%g, %g, %g, %g, %g, %g]]}"
                % tuple(self.toarray().ravel()))

    def __array__(self, dtype=None):
        return self.toarray().astype(dtype)

    def __getitem__(self, tuple indices):
        cdef int i, j
        i, j = indices
        if i < 0 or i >= 6:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 6:
            raise KeyError("index out of range %d" % j)
        return self.thisptr.arr[i].arr[j]

    def __setitem__(self, tuple indices, double v):
        cdef int i, j
        i, j = indices
        if i < 0 or i >= 6:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 6:
            raise KeyError("index out of range %d" % j)
        self.thisptr.arr[i].arr[j] = v

    @property
    def shape(self):
        return (6, 6)

    def assign(self, Matrix6d other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=2] array = np.empty((6, 6))
        cdef int i, j
        for i in range(6):
            for j in range(6):
                array[i, j] = self.thisptr.arr[i].arr[j]
        return array

    def fromarray(self, np.ndarray[double, ndim=2] array):
        cdef int i, j
        for i in range(6):
            for j in range(6):
                self.thisptr.arr[i].arr[j] = array[i, j]


cdef class Quaterniond:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.thisptr = new _cdff_types.asn1SccQuaterniond()
        self.delete_thisptr = True
        self.thisptr.arr[0] = x
        self.thisptr.arr[1] = y
        self.thisptr.arr[2] = z
        self.thisptr.arr[3] = w

    def __len__(self):
        return 4

    def __str__(self):
        return str("{type: Quaterniond, data: {x: %g, y: %g, z: %g, w: %g}}"
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

    def assign(self, Quaterniond other):
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

    def conjugate(self):
        conj = Quaterniond()
        conj[0] = -self.thisptr.arr[0]
        conj[1] = -self.thisptr.arr[1]
        conj[2] = -self.thisptr.arr[2]
        conj[3] = self.thisptr.arr[3]
        return conj

    def __mul__(self, Quaterniond other):
        cdef Quaterniond result = Quaterniond()
        result[0] = self[3] * other[0] + other[3] * self[0] + self[1] * other[2] - self[2] * other[1]
        result[1] = self[3] * other[1] + other[3] * self[1] + self[2] * other[0] - self[0] * other[2]
        result[2] = self[3] * other[2] + other[3] * self[2] + self[0] * other[1] - self[1] * other[0]
        result[3] = self[3] * other[3] - self[0] * other[0] - self[1] * other[1] - self[2] * other[2]
        return result


cdef class Pose:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccPose()
        self.delete_thisptr = True

    def __str__(self):
        return "{type: Pose, pos: %s, orient: %s}" % (self.pos, self.orient)

    @property
    def pos(self):
        cdef Vector3d pos = Vector3d()
        del pos.thisptr
        pos.delete_thisptr = False
        pos.thisptr = &self.thisptr.pos
        return pos

    @property
    def orient(self):
        cdef Quaterniond orient = Quaterniond()
        del orient.thisptr
        orient.delete_thisptr = False
        orient.thisptr = &self.thisptr.orient
        return orient


cdef class TransformWithCovariance_MetadataReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return ("{msg_version: %d, producer_id: %s, parent_frame_id: %s, "
                "parent_time: %s, child_frame_id: %s, child_time: %s}"
                % (self.thisptr.msgVersion, self.producer_id,
                   self.parent_frame_id, self.parent_time, self.child_frame_id,
                   self.child_time))

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    def _get_producer_id(self):
        cdef bytes producer_id = self.thisptr.producerId.arr
        return producer_id.decode()

    def _set_producer_id(self, str producer_id):
        cdef string value = producer_id.encode()
        memcpy(self.thisptr.producerId.arr, value.c_str(), len(producer_id))
        self.thisptr.producerId.nCount = len(producer_id)

    producer_id = property(_get_producer_id, _set_producer_id)

    # TODO asn1SccTransformWithCovariance_Metadata_dataEstimated dataEstimated

    def _get_parent_frame_id(self):
        cdef bytes parent_frame_id = self.thisptr.parentFrameId.arr
        return parent_frame_id.decode()

    def _set_parent_frame_id(self, str parent_frame_id):
        cdef string value = parent_frame_id.encode()
        memcpy(self.thisptr.parentFrameId.arr, value.c_str(), len(parent_frame_id))
        self.thisptr.parentFrameId.nCount = len(parent_frame_id)

    parent_frame_id = property(_get_parent_frame_id, _set_parent_frame_id)

    def _get_parent_time(self):
        cdef Time parent_time = Time()
        del parent_time.thisptr
        parent_time.thisptr = &self.thisptr.parentTime
        parent_time.delete_thisptr = False
        return parent_time

    def _set_parent_time(self, Time parent_time):
        self.thisptr.parentTime = deref(parent_time.thisptr)

    parent_time = property(_get_parent_time, _set_parent_time)

    def _get_child_frame_id(self):
        cdef bytes child_frame_id = self.thisptr.childFrameId.arr
        return child_frame_id.decode()

    def _set_child_frame_id(self, str child_frame_id):
        cdef string value = child_frame_id.encode()
        memcpy(self.thisptr.childFrameId.arr, value.c_str(), len(child_frame_id))
        self.thisptr.childFrameId.nCount = len(child_frame_id)

    child_frame_id = property(_get_child_frame_id, _set_child_frame_id)

    def _get_child_time(self):
        cdef Time child_time = Time()
        del child_time.thisptr
        child_time.thisptr = &self.thisptr.childTime
        child_time.delete_thisptr = False
        return child_time

    def _set_child_time(self, Time child_time):
        self.thisptr.childTime = deref(child_time.thisptr)

    child_time = property(_get_child_time, _set_child_time)


cdef class TransformWithCovariance_DataReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return ("{translation: %s, orientation: %s, cov: %s}"
                % (self.translation, self.orientation, self.cov))

    @property
    def translation(self):
        cdef Vector3d translation = Vector3d()
        del translation.thisptr
        translation.delete_thisptr = False
        translation.thisptr = &self.thisptr.translation
        return translation

    @property
    def orientation(self):
        cdef Quaterniond orientation = Quaterniond()
        del orientation.thisptr
        orientation.delete_thisptr = False
        orientation.thisptr = &self.thisptr.orientation
        return orientation

    @property
    def cov(self):
        cdef Matrix6d cov = Matrix6d()
        del cov.thisptr
        cov.delete_thisptr = False
        cov.thisptr = &self.thisptr.cov
        return cov


cdef class TransformWithCovariance:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccTransformWithCovariance()
        self.delete_thisptr = True

    def __str__(self):
        return ("{metadata: %s, data: %s}"
                % (self.metadata, self.data))

    @property
    def metadata(self):
        cdef TransformWithCovariance_MetadataReference metadata = \
            TransformWithCovariance_MetadataReference()
        metadata.thisptr = &self.thisptr.metadata
        return metadata

    @property
    def data(self):
        cdef TransformWithCovariance_DataReference data = \
            TransformWithCovariance_DataReference()
        data.thisptr = &self.thisptr.data
        return data

    def from_uper(self, list data):
        cdef unsigned char* uper_buffer = <unsigned char*> malloc(
            _cdff_types.asn1SccTransformWithCovariance_REQUIRED_BYTES_FOR_ENCODING *
            sizeof(unsigned char))
        cdef _cdff_types.BitStream* b = new _cdff_types.BitStream()
        cdef int i
        _cdff_types.BitStream_Init(
            b, uper_buffer, _cdff_types.asn1SccTransformWithCovariance_REQUIRED_BYTES_FOR_ENCODING)
        for i in range(_cdff_types.asn1SccTransformWithCovariance_REQUIRED_BYTES_FOR_ENCODING):
            uper_buffer[i] = data[i]
            b.buf[i] = data[i]
        b.count = _cdff_types.asn1SccTransformWithCovariance_REQUIRED_BYTES_FOR_ENCODING

        cdef int error_code
        cdef bool success
        success = _cdff_types.asn1SccTransformWithCovariance_Decode(self.thisptr, b, &error_code)
        if not success:
            raise RuntimeError(
                "Conversion from uPER failed, error code: %d" % error_code)

        free(uper_buffer)
        del b


cdef class PointCloud_Data_pointsReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __getitem__(self, tuple indices):
        cdef int i, j
        i, j = indices

        if i < 0 or i >= self.thisptr.nCount:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 3:
            raise KeyError("index out of range %d" % j)

        return self.thisptr.arr[i].arr[j]

    def __setitem__(self, tuple indices, double v):
        cdef int i, j, k
        i, j = indices

        if self.thisptr.nCount <= i:
            self.thisptr.nCount = i + 1

        if i < 0 or i >= self.thisptr.nCount:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 3:
            raise KeyError("index out of range %d" % j)
        self.thisptr.arr[i].arr[j] = v

    @property
    def shape(self):
        return (self.thisptr.nCount, 3)

    def resize(self, int size):
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount

    def fill(self, list data):
        cdef int data_len = len(data)
        self.resize(data_len)

        cdef int i, j
        for i in range(data_len):
            for j in range(3):
                self.thisptr.arr[i].arr[j] = data[i][j]


cdef class PointCloud_Data_colorsReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __getitem__(self, tuple indices):
        cdef int i, j
        i, j = indices

        if i < 0 or i >= self.thisptr.nCount:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 3:
            raise KeyError("index out of range %d" % j)

        return self.thisptr.arr[i].arr[j]

    def __setitem__(self, tuple indices, double v):
        cdef int i, j, k
        i, j = indices

        if self.thisptr.nCount <= i:
            self.thisptr.nCount = i + 1

        if i < 0 or i >= self.thisptr.nCount:
            raise KeyError("index out of range %d" % i)
        if j < 0 or j >= 3:
            raise KeyError("index out of range %d" % j)
        self.thisptr.arr[i].arr[j] = v

    @property
    def shape(self):
        return (self.thisptr.nCount, 3)

    def resize(self, int size):
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount

    def fill(self, list data):
        cdef int data_len = len(data)
        self.resize(data_len)

        cdef int i, j
        for i in range(data_len):
            for j in range(3):
                self.thisptr.arr[i].arr[j] = data[i][j]


cdef class PointCloud_Data_intensityReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __getitem__(self, int i):
        return self.thisptr.arr[i]

    def __setitem__(self, int i, int32_t v):
        if i >= 400000:
            warnings.warn("Maximum size of Pointcloud is %d" % 400000)
            return
        if self.thisptr.nCount <= i:
            self.thisptr.nCount = i + 1
        self.thisptr.arr[i] = v

    def __len__(self):
        return self.thisptr.nCount

    def resize(self, int size):
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount

    def fill(self, list data):
        cdef int data_len = len(data)
        self.resize(data_len)

        cdef int i
        for i in range(data_len):
            self.thisptr.arr[i] = data[i]


cdef class PointCloud_DataReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return ("{points: %d, colors: %d, intensity: %d}"
                % (self.points.size(), self.colors.size(),
                   self.intensity.size()))

    @property
    def points(self):
        cdef PointCloud_Data_pointsReference points = \
            PointCloud_Data_pointsReference()
        points.thisptr = &self.thisptr.points
        return points

    @property
    def colors(self):
        cdef PointCloud_Data_colorsReference colors = \
            PointCloud_Data_colorsReference()
        colors.thisptr = &self.thisptr.colors
        return colors

    @property
    def intensity(self):
        cdef PointCloud_Data_intensityReference intensity = \
            PointCloud_Data_intensityReference()
        intensity.thisptr = &self.thisptr.intensity
        return intensity


cdef class PointCloud_MetadataReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return ("{msg_version: %d, time_stamp: %s, sensor_id: %s, frame_id: %s,"
                " height: %d, width: %d, is_registered: %s, is_ordered: %s, "
                "has_fixed_transform: %s, pose_fixed_frame_robot_frame: %s}"
                % (self.msg_version, self.time_stamp, self.sensor_id,
                   self.frame_id, self.height, self.width, self.is_registered,
                   self.is_ordered, self.has_fixed_transform,
                   self.pose_fixed_frame_robot_frame))

    def _get_time_stamp(self):
        cdef Time time = Time()
        del time.thisptr
        time.thisptr = &self.thisptr.timeStamp
        time.delete_thisptr = False
        return time

    def _set_time_stamp(self, Time time):
        self.thisptr.timeStamp = deref(time.thisptr)

    time_stamp = property(_get_time_stamp, _set_time_stamp)

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    def _get_sensor_id(self):
        cdef bytes sensor_id = self.thisptr.sensorId.arr
        return sensor_id.decode()

    def _set_sensor_id(self, str sensor_id):
        cdef string value = sensor_id.encode()
        memcpy(self.thisptr.sensorId.arr, value.c_str(), len(sensor_id))
        self.thisptr.sensorId.nCount = len(sensor_id)

    sensor_id = property(_get_sensor_id, _set_sensor_id)

    def _get_frame_id(self):
        cdef bytes frame_id = self.thisptr.frameId.arr
        return frame_id.decode()

    def _set_frame_id(self, str frame_id):
        cdef string value = frame_id.encode()
        memcpy(self.thisptr.frameId.arr, value.c_str(), len(frame_id))
        self.thisptr.frameId.nCount = len(frame_id)

    frame_id = property(_get_frame_id, _set_frame_id)

    def _get_height(self):
        return self.thisptr.height

    def _set_height(self, uint32_t height):
        self.thisptr.height = height

    height = property(_get_height, _set_height)

    def _get_width(self):
        return self.thisptr.width

    def _set_width(self, uint32_t width):
        self.thisptr.width = width

    width = property(_get_width, _set_width)

    def _get_is_registered(self):
        return self.thisptr.isRegistered

    def _set_is_registered(self, bool is_registered):
        self.thisptr.isRegistered = is_registered

    is_registered = property(_get_is_registered, _set_is_registered)

    def _get_is_ordered(self):
        return self.thisptr.isOrdered

    def _set_is_ordered(self, bool is_ordered):
        self.thisptr.isOrdered = is_ordered

    is_ordered = property(_get_is_ordered, _set_is_ordered)

    def _get_has_fixed_transform(self):
        return self.thisptr.hasFixedTransform

    def _set_has_fixed_transform(self, bool has_fixed_transform):
        self.thisptr.hasFixedTransform = has_fixed_transform

    has_fixed_transform = property(
        _get_has_fixed_transform, _set_has_fixed_transform)

    @property
    def pose_robot_frame_sensor_frame(self):
        cdef TransformWithCovariance pose_robotFrame_sensorFrame = \
            TransformWithCovariance()
        del pose_robotFrame_sensorFrame.thisptr
        pose_robotFrame_sensorFrame.delete_thisptr = False
        pose_robotFrame_sensorFrame.thisptr = \
            &self.thisptr.pose_robotFrame_sensorFrame
        return pose_robotFrame_sensorFrame

    @property
    def pose_fixed_frame_robot_frame(self):
        cdef TransformWithCovariance pose_fixedFrame_robotFrame = \
            TransformWithCovariance()
        del pose_fixedFrame_robotFrame.thisptr
        pose_fixedFrame_robotFrame.delete_thisptr = False
        pose_fixedFrame_robotFrame.thisptr = \
            &self.thisptr.pose_fixedFrame_robotFrame
        return pose_fixedFrame_robotFrame


cdef class Pointcloud:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccPointcloud()
        self.delete_thisptr = True
        _cdff_types.asn1SccPointcloud_Initialize(self.thisptr)
        self.thisptr.metadata.msgVersion = _cdff_types.pointCloud_Version

    def __str__(self):
        return ("{type: Pointcloud, metadata: %s, data: %s}"
                % (self.metadata, self.data))

    @property
    def metadata(self):
        cdef PointCloud_MetadataReference metadata = \
            PointCloud_MetadataReference()
        metadata.thisptr = &self.thisptr.metadata
        return metadata

    @property
    def data(self):
        cdef PointCloud_DataReference data = PointCloud_DataReference()
        data.thisptr = &self.thisptr.data
        return data

    def from_uper(self, list data):
        cdef unsigned char* uper_buffer = <unsigned char*> malloc(
            _cdff_types.asn1SccPointcloud_REQUIRED_BYTES_FOR_ENCODING *
            sizeof(unsigned char))
        cdef _cdff_types.BitStream* b = new _cdff_types.BitStream()
        cdef int i
        _cdff_types.BitStream_Init(
            b, uper_buffer, _cdff_types.asn1SccPointcloud_REQUIRED_BYTES_FOR_ENCODING)
        for i in range(_cdff_types.asn1SccPointcloud_REQUIRED_BYTES_FOR_ENCODING):
            uper_buffer[i] = data[i]
            b.buf[i] = data[i]
        b.count = _cdff_types.asn1SccPointcloud_REQUIRED_BYTES_FOR_ENCODING

        cdef int error_code
        cdef bool success
        success = _cdff_types.asn1SccPointcloud_Decode(self.thisptr, b, &error_code)
        if not success:
            raise RuntimeError(
                "Conversion from uPER failed, error code: %d" % error_code)

        free(uper_buffer)
        del b


cdef class LaserScan:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccLaserScan()
        self.thisptr.ranges.nCount = 0
        self.thisptr.remission.nCount = 0
        self.thisptr.start_angle = 0.0
        self.thisptr.angular_resolution = 0.0
        self.thisptr.speed = 0.0
        self.thisptr.minRange = 0
        self.thisptr.maxRange = 0
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.ranges.nCount

    def __str__(self):
        return str(
            "{type: LaserScan, start_angle=%g, angular_resolution=%g, "
            "speed=%g, min_range=%g, max_range=%g, ranges=%s, remission=%s}"
            % (self.thisptr.start_angle, self.thisptr.angular_resolution,
               self.thisptr.speed, self.thisptr.minRange, self.thisptr.maxRange,
               self.ranges, self.remission))

    def assign(self, LaserScan other):
        self.thisptr.assign(deref(other.thisptr))

    def _get_ref_time(self):
        cdef Time time = Time()
        del time.thisptr
        time.thisptr = &self.thisptr.ref_time
        time.delete_thisptr = False
        return time

    def _set_ref_time(self, Time time):
        self.thisptr.ref_time = deref(time.thisptr)

    ref_time = property(_get_ref_time, _set_ref_time)

    def _get_start_angle(self):
        return self.thisptr.start_angle

    def _set_start_angle(self, double start_angle):
        self.thisptr.start_angle = start_angle

    start_angle = property(_get_start_angle, _set_start_angle)

    def _get_angular_resolution(self):
        return self.thisptr.angular_resolution

    def _set_angular_resolution(self, double angular_resolution):
        self.thisptr.angular_resolution = angular_resolution

    angular_resolution = property(_get_angular_resolution, _set_angular_resolution)

    def _get_speed(self):
        return self.thisptr.speed

    def _set_speed(self, double speed):
        self.thisptr.speed = speed

    speed = property(_get_speed, _set_speed)

    def _get_min_range(self):
        return self.thisptr.minRange

    def _set_min_range(self, uint32_t min_range):
        self.thisptr.minRange = min_range

    min_range = property(_get_min_range, _set_min_range)

    def _get_max_range(self):
        return self.thisptr.maxRange

    def _set_max_range(self, uint32_t max_range):
        self.thisptr.maxRange = max_range

    max_range = property(_get_max_range, _set_max_range)

    @property
    def remission(self):
        cdef LaserScan_remissionReference remission = LaserScan_remissionReference()
        remission.thisptr = &self.thisptr.remission
        return remission

    @property
    def ranges(self):
        cdef LaserScan_rangesReference ranges = LaserScan_rangesReference()
        ranges.thisptr = &self.thisptr.ranges
        return ranges


cdef class LaserScan_remissionReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __getitem__(self, uint32_t i):
        return self.thisptr.arr[i]

    def __setitem__(self, uint32_t i, float v):
        if i >= 2000:
            warnings.warn("Maximum size of LaserScan is %d" % 2000)
            return
        self.thisptr.arr[i] = v
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, uint32_t size):
        if size > 2000:
            warnings.warn("Maximum size of LaserScan is %d" % 2000)
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("[%s]" % (", ".join(str(self.thisptr.arr[i])
                                       for i in range(self.size()))))


cdef class LaserScan_rangesReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __getitem__(self, uint32_t i):
        return self.thisptr.arr[i]

    def __setitem__(self, uint32_t i, uint32_t v):
        if i >= 2000:
            warnings.warn("Maximum size of LaserScan is %d" % 2000)
            return
        self.thisptr.arr[i] = v
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, uint32_t size):
        if size > 2000:
            warnings.warn("Maximum size of LaserScan is %d" % 2000)
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("[%s]" % (", ".join(str(self.thisptr.arr[i])
                                       for i in range(self.size()))))


cdef class RigidBodyState:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccRigidBodyState()
        self.delete_thisptr = True

    def __str__(self):
        return str(
            "{type: RigidBodyState, timestamp=%s, sourceFrame=%s, "
            "targetFrame=%s, pos=%s, orient=%s, ...}"
            % (self.timestamp, self.source_frame, self.target_frame, self.pos,
               self.orient))

    def _get_timestamp(self):
        cdef Time timestamp = Time()
        del timestamp.thisptr
        timestamp.thisptr = &self.thisptr.timestamp
        timestamp.delete_thisptr = False
        return timestamp

    def _set_timestamp(self, Time timestamp):
        self.thisptr.timestamp = deref(timestamp.thisptr)

    timestamp = property(_get_timestamp, _set_timestamp)

    def _get_source_frame(self):
        cdef bytes source_frame = self.thisptr.sourceFrame.arr
        return source_frame.decode()

    def _set_source_frame(self, str sourceFrame):
        cdef string value = sourceFrame.encode()
        memcpy(self.thisptr.sourceFrame.arr, value.c_str(), len(sourceFrame))
        self.thisptr.sourceFrame.nCount = len(sourceFrame)

    source_frame = property(_get_source_frame, _set_source_frame)

    def _get_target_frame(self):
        cdef bytes target_frame = self.thisptr.targetFrame.arr
        return target_frame.decode()

    def _set_target_frame(self, str targetFrame):
        cdef string value = targetFrame.encode()
        memcpy(self.thisptr.targetFrame.arr, value.c_str(), len(targetFrame))
        self.thisptr.targetFrame.nCount = len(targetFrame)

    target_frame = property(_get_target_frame, _set_target_frame)

    def _get_pos(self):
        cdef Vector3d pos = Vector3d()
        del pos.thisptr
        pos.delete_thisptr = False
        pos.thisptr = &self.thisptr.pos
        return pos

    def _set_pos(self, Vector3d value):
        self.thisptr.pos = deref(value.thisptr)

    pos = property(_get_pos, _set_pos)

    def _get_cov_position(self):
        cdef Matrix3d cov_position = Matrix3d()
        del cov_position.thisptr
        cov_position.delete_thisptr = False
        cov_position.thisptr = &self.thisptr.cov_position
        return cov_position

    def _set_cov_position(self, Matrix3d value):
        self.thisptr.cov_position = deref(value.thisptr)

    cov_position = property(_get_cov_position, _set_cov_position)

    def _get_orient(self):
        cdef Quaterniond orient = Quaterniond()
        del orient.thisptr
        orient.delete_thisptr = False
        orient.thisptr = &self.thisptr.orient
        return orient

    def _set_orient(self, Quaterniond value):
        self.thisptr.orient = deref(value.thisptr)

    orient = property(_get_orient, _set_orient)

    def _get_cov_orientation(self):
        cdef Matrix3d cov_orientation = Matrix3d()
        del cov_orientation.thisptr
        cov_orientation.delete_thisptr = False
        cov_orientation.thisptr = &self.thisptr.cov_orientation
        return cov_orientation

    def _set_cov_orientation(self, Matrix3d value):
        self.thisptr.cov_orientation = deref(value.thisptr)

    cov_orientation = property(_get_cov_orientation, _set_cov_orientation)

    def _get_velocity(self):
        cdef Vector3d velocity = Vector3d()
        del velocity.thisptr
        velocity.delete_thisptr = False
        velocity.thisptr = &self.thisptr.velocity
        return velocity

    def _set_velocity(self, Vector3d value):
        self.thisptr.velocity = deref(value.thisptr)

    velocity = property(_get_velocity, _set_velocity)

    def _get_cov_velocity(self):
        cdef Matrix3d cov_velocity = Matrix3d()
        del cov_velocity.thisptr
        cov_velocity.delete_thisptr = False
        cov_velocity.thisptr = &self.thisptr.cov_velocity
        return cov_velocity

    def _set_cov_velocity(self, Matrix3d value):
        self.thisptr.cov_velocity = deref(value.thisptr)

    cov_velocity = property(_get_cov_velocity, _set_cov_velocity)

    def _get_angular_velocity(self):
        cdef Vector3d angular_velocity = Vector3d()
        del angular_velocity.thisptr
        angular_velocity.delete_thisptr = False
        angular_velocity.thisptr = &self.thisptr.angular_velocity
        return angular_velocity

    def _set_angular_velocity(self, Vector3d value):
        self.thisptr.angular_velocity = deref(value.thisptr)

    angular_velocity = property(_get_angular_velocity, _set_angular_velocity)

    def _get_cov_angular_velocity(self):
        cdef Matrix3d cov_angular_velocity = Matrix3d()
        del cov_angular_velocity.thisptr
        cov_angular_velocity.delete_thisptr = False
        cov_angular_velocity.thisptr = &self.thisptr.cov_angular_velocity
        return cov_angular_velocity

    def _set_cov_angular_velocity(self, Matrix3d value):
        self.thisptr.cov_angular_velocity = deref(value.thisptr)

    cov_angular_velocity = property(
        _get_cov_angular_velocity, _set_cov_angular_velocity)


cdef class JointState:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccJointState()
        self.delete_thisptr = True

    def __str__(self):
        return str(
            "{type: JointState, position: %g, speed: %g, "
            "effort: %g, raw: %g, acceleration: %g}"
            % (self.thisptr.position, self.thisptr.speed,
               self.thisptr.effort, self.thisptr.raw,
               self.thisptr.acceleration))

    def assign(self, JointState other):
        self.thisptr.assign(deref(other.thisptr))

    def _get_position(self):
        return self.thisptr.position

    def _set_position(self, double position):
        self.thisptr.position = position

    position = property(_get_position, _set_position)

    def _get_speed(self):
        return self.thisptr.speed

    def _set_speed(self, float speed):
        self.thisptr.speed = speed

    speed = property(_get_speed, _set_speed)

    def _get_effort(self):
        return self.thisptr.effort

    def _set_effort(self, float effort):
        self.thisptr.effort = effort

    effort = property(_get_effort, _set_effort)

    def _get_raw(self):
        return self.thisptr.raw

    def _set_raw(self, float raw):
        self.thisptr.raw = raw

    raw = property(_get_raw, _set_raw)

    def _get_acceleration(self):
        return self.thisptr.acceleration

    def _set_acceleration(self, float acceleration):
        self.thisptr.acceleration = acceleration

    acceleration = property(_get_acceleration, _set_acceleration)


cdef class Joints:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccJoints()
        self.delete_thisptr = True

    def __str__(self):
        return str("{type: Joints, ...}") # TODO

    def assign(self, Joints other):
        self.thisptr.assign(deref(other.thisptr))

    def _get_timestamp(self):
        cdef Time timestamp = Time()
        del timestamp.thisptr
        timestamp.thisptr = &self.thisptr.timestamp
        timestamp.delete_thisptr = False
        return timestamp

    def _set_timestamp(self, Time timestamp):
        self.thisptr.timestamp = deref(timestamp.thisptr)

    timestamp = property(_get_timestamp, _set_timestamp)

    @property
    def names(self):
        cdef Joints_namesReference names = Joints_namesReference()
        names.thisptr = &self.thisptr.names
        return names

    @property
    def elements(self):
        cdef Joints_elementsReference elements = Joints_elementsReference()
        elements.thisptr = &self.thisptr.elements
        return elements


cdef class Joints_namesReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __getitem__(self, int i):
        cdef bytes name = self.thisptr.arr[i].arr
        return name.decode()

    def __setitem__(self, int i, str name):
        if i >= 30:
            warnings.warn("Maximum size of Joints is 30")
            return
        cdef string value = name.encode()
        memcpy(self.thisptr.arr[i].arr, value.c_str(), len(name))
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, int size):
        if size > 30:
            warnings.warn("Maximum size of Joints is 30")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class Joints_elementsReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        cdef JointState joint_state = JointState()
        joint_state.delete_thisptr = False
        del joint_state.thisptr
        joint_state.thisptr = &self.thisptr.arr[i]
        return joint_state

    def __setitem__(self, int i, JointState joint_state):
        if i >= 30:
            warnings.warn("Maximum size of Joints is 30")
            return
        self.thisptr.arr[i] = deref(joint_state.thisptr)
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, int size):
        if size > 30:
            warnings.warn("Maximum size of Joints is 30")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class IMUSensors:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccIMUSensors()
        self.delete_thisptr = True

    def __str__(self):
        return str(
            "{type: IMUSensors, timestamp=%s, acc=%s, gyro=%s, mag=%s}"
            % (self.timestamp, self.acc, self.gyro, self.mag))

    def _get_timestamp(self):
        cdef Time timestamp = Time()
        del timestamp.thisptr
        timestamp.thisptr = &self.thisptr.timestamp
        timestamp.delete_thisptr = False
        return timestamp

    def _set_timestamp(self, Time timestamp):
        self.thisptr.timestamp = deref(timestamp.thisptr)

    timestamp = property(_get_timestamp, _set_timestamp)

    def _get_acc(self):
        cdef Vector3d acc = Vector3d()
        del acc.thisptr
        acc.delete_thisptr = False
        acc.thisptr = &self.thisptr.acc
        return acc

    def _set_acc(self, Vector3d value):
        self.thisptr.acc = deref(value.thisptr)

    acc = property(_get_acc, _set_acc)

    def _get_gyro(self):
        cdef Vector3d gyro = Vector3d()
        del gyro.thisptr
        gyro.delete_thisptr = False
        gyro.thisptr = &self.thisptr.gyro
        return gyro

    def _set_gyro(self, Vector3d value):
        self.thisptr.gyro = deref(value.thisptr)

    gyro = property(_get_gyro, _set_gyro)

    def _get_mag(self):
        cdef Vector3d mag = Vector3d()
        del mag.thisptr
        mag.delete_thisptr = False
        mag.thisptr = &self.thisptr.mag
        return mag

    def _set_mag(self, Vector3d value):
        self.thisptr.mag = deref(value.thisptr)

    mag = property(_get_mag, _set_mag)


cdef class DepthMap:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccDepthMap()
        self.delete_thisptr = True

    def __str__(self):
        # TODO
        return str(
            "{type: DepthMap, ref_time=%s, ...}"
            % (self.ref_time,))

    def _get_ref_time(self):
        cdef Time ref_time = Time()
        del ref_time.thisptr
        ref_time.thisptr = &self.thisptr.ref_time
        ref_time.delete_thisptr = False
        return ref_time

    def _set_ref_time(self, Time ref_time):
        self.thisptr.ref_time = deref(ref_time.thisptr)

    ref_time = property(_get_ref_time, _set_ref_time)

    @property
    def timestamps(self):
        cdef DepthMap_timestampsReference timestamps = DepthMap_timestampsReference()
        timestamps.thisptr = &self.thisptr.timestamps
        return timestamps

    def _get_vertical_projection(self):
        if <int> self.thisptr.vertical_projection == <int> _cdff_types.asn1Sccpolar:
            return "polar"
        else:
            return "planar"

    def _set_vertical_projection(self,  str vertical_projection):
        if vertical_projection == "polar":
            self.thisptr.vertical_projection = _cdff_types.asn1Sccpolar
        elif vertical_projection == "planar":
            self.thisptr.vertical_projection = _cdff_types.asn1Sccplanar
        else:
            raise ValueError("Unknown projection: %s" % vertical_projection)

    vertical_projection = property(_get_vertical_projection, _set_vertical_projection)

    def _get_horizontal_projection(self):
        if <int> self.thisptr.horizontal_projection == <int> _cdff_types.asn1Sccpolar:
            return "polar"
        else:
            return "planar"

    def _set_horizontal_projection(self,  str horizontal_projection):
        if horizontal_projection == "polar":
            self.thisptr.horizontal_projection = _cdff_types.asn1Sccpolar
        elif horizontal_projection == "planar":
            self.thisptr.horizontal_projection = _cdff_types.asn1Sccplanar
        else:
            raise ValueError("Unknown projection: %s" % horizontal_projection)

    horizontal_projection = property(_get_horizontal_projection, _set_horizontal_projection)

    @property
    def vertical_interval(self):
        cdef DepthMap_vertical_intervalReference vertical_interval = DepthMap_vertical_intervalReference()
        vertical_interval.thisptr = &self.thisptr.vertical_interval
        return vertical_interval

    @property
    def horizontal_interval(self):
        cdef DepthMap_horizontal_intervalReference horizontal_interval = DepthMap_horizontal_intervalReference()
        horizontal_interval.thisptr = &self.thisptr.horizontal_interval
        return horizontal_interval

    def _get_vertical_size(self):
        return self.thisptr.vertical_size

    def _set_vertical_size(self,  uint32_t vertical_size):
        self.thisptr.vertical_size = vertical_size

    vertical_size = property(_get_vertical_size, _set_vertical_size)

    def _get_horizontal_size(self):
        return self.thisptr.horizontal_size

    def _set_horizontal_size(self,  uint32_t horizontal_size):
        self.thisptr.horizontal_size = horizontal_size

    horizontal_size = property(_get_horizontal_size, _set_horizontal_size)

    @property
    def distances(self):
        cdef DepthMap_distancesReference distances = DepthMap_distancesReference()
        distances.thisptr = &self.thisptr.distances
        return distances

    @property
    def remissions(self):
        cdef DepthMap_remissionsReference remissions = DepthMap_remissionsReference()
        remissions.thisptr = &self.thisptr.remissions
        return remissions


cdef class DepthMap_timestampsReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        cdef Time time = Time()
        time.delete_thisptr = False
        del time.thisptr
        time.thisptr = &self.thisptr.arr[i]
        return time

    def __setitem__(self, int i, Time timestamp):
        if i >= 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.arr[i] = deref(timestamp.thisptr)
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, int size):
        if size > 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


class ProjectionType:
    POLAR = <int> _cdff_types.asn1Sccpolar
    PLANAR = <int> _cdff_types.asn1Sccplanar


cdef class DepthMap_vertical_intervalReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        return self.thisptr.arr[i]

    def __setitem__(self, int i, double v):
        if i >= 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.arr[i] = v
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, int size):
        if size > 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class DepthMap_horizontal_intervalReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        return self.thisptr.arr[i]

    def __setitem__(self, int i, double v):
        if i >= 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.arr[i] = v
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, int size):
        if size > 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class DepthMap_distancesReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        return self.thisptr.arr[i]

    def __setitem__(self, int i, double v):
        if i >= 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.arr[i] = v
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, int size):
        if size > 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class DepthMap_remissionsReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        return self.thisptr.arr[i]

    def __setitem__(self, int i, double v):
        if i >= 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.arr[i] = v
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, int size):
        if size > 30000:
            warnings.warn("Maximum size of DepthMap is 30000")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class Image:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccImage()
        self.thisptr.image.nCount = 0
        self.thisptr.attributes.nCount = 0
        self.thisptr.datasize.width = 0
        self.thisptr.datasize.height = 0
        self.thisptr.data_depth = 0
        self.thisptr.pixel_size = 0
        self.thisptr.row_size = 0

        self.delete_thisptr = True

    def __str__(self):
        # TODO
        return str("{type: Image}")

    def _get_frame_time(self):
        cdef Time frame_time = Time()
        del frame_time.thisptr
        frame_time.thisptr = &self.thisptr.frame_time
        frame_time.delete_thisptr = False
        return frame_time

    def _set_frame_time(self, Time frame_time):
        self.thisptr.frame_time = deref(frame_time.thisptr)

    frame_time = property(_get_frame_time, _set_frame_time)

    def _get_received_time(self):
        cdef Time received_time = Time()
        del received_time.thisptr
        received_time.thisptr = &self.thisptr.received_time
        received_time.delete_thisptr = False
        return received_time

    def _set_received_time(self, Time received_time):
        self.thisptr.received_time = deref(received_time.thisptr)

    received_time = property(_get_received_time, _set_received_time)

    @property
    def image(self):
        cdef Image_imageReference image = Image_imageReference()
        image.thisptr = &self.thisptr.image
        return image

    @property
    def attributes(self):
        cdef Image_attributesReference attributes = Image_attributesReference()
        attributes.thisptr = &self.thisptr.attributes
        return attributes

    def _get_datasize(self):
        cdef Image_size_tReference datasize = Image_size_tReference()
        del datasize.thisptr
        datasize.thisptr = &self.thisptr.datasize
        return datasize

    def _set_datasize(self, Image_size_tReference datasize):
        self.thisptr.datasize = deref(datasize.thisptr)

    datasize = property(_get_datasize, _set_datasize)

    def _get_data_depth(self):
        return self.thisptr.data_depth

    def _set_data_depth(self, uint32_t data_depth):
        self.thisptr.data_depth = data_depth

    data_depth = property(_get_data_depth, _set_data_depth)

    def _get_pixel_size(self):
        return self.thisptr.pixel_size

    def _set_pixel_size(self, uint32_t pixel_size):
        self.thisptr.pixel_size = pixel_size

    pixel_size = property(_get_pixel_size, _set_pixel_size)

    def _get_row_size(self):
        return self.thisptr.row_size

    def _set_row_size(self, uint32_t row_size):
        self.thisptr.row_size = row_size

    row_size = property(_get_row_size, _set_row_size)

    def _get_frame_mode(self):
        if <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_undefined:
            return "mode_undefined"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_grayscale:
            return "mode_grayscale"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_rgb:
            return "mode_rgb"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_uyvy:
            return "mode_uyvy"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_bgr:
            return "mode_bgr"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_rgb32:
            return "mode_rgb32"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccraw_modes:
            return "raw_modes"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_bayer:
            return "mode_bayer"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_bayer_rggb:
            return "mode_bayer_rggb"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_bayer_grbg:
            return "mode_bayer_grbg"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_bayer_bggr:
            return "mode_bayer_bggr"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_bayer_gbrg:
            return "mode_bayer_gbrg"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Scccompressed_modes:
            return "compressed_modes"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1SccImage_mode_t_mode_pjpg:
            return "Image_mode_t_mode_pjpg"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_jpeg:
            return "mode_jpeg"
        elif <int> self.thisptr.frame_mode == <int> _cdff_types.asn1Sccmode_png:
            return "mode_png"
        else:
            raise ValueError("Unknown frame_mode: %s" % <int> self.thisptr.frame_mode)

    def _set_frame_mode(self,  str frame_mode):
        if frame_mode == "mode_undefined":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_undefined
        elif frame_mode == "mode_grayscale":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_grayscale
        elif frame_mode == "mode_rgb":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_rgb
        elif frame_mode == "mode_uyvy":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_uyvy
        elif frame_mode == "mode_bgr":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_bgr
        elif frame_mode == "mode_rgb32":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_rgb32
        elif frame_mode == "raw_modes":
            self.thisptr.frame_mode = _cdff_types.asn1Sccraw_modes
        elif frame_mode == "mode_bayer":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_bayer
        elif frame_mode == "mode_bayer_rggb":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_bayer_rggb
        elif frame_mode == "mode_bayer_grbg":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_bayer_grbg
        elif frame_mode == "mode_bayer_bggr":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_bayer_bggr
        elif frame_mode == "mode_bayer_gbrg":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_bayer_gbrg
        elif frame_mode == "compressed_modes":
            self.thisptr.frame_mode = _cdff_types.asn1Scccompressed_modes
        elif frame_mode == "Image_mode_t_mode_pjpg":
            self.thisptr.frame_mode = _cdff_types.asn1SccImage_mode_t_mode_pjpg
        elif frame_mode == "mode_jpeg":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_jpeg
        elif frame_mode == "mode_png":
            self.thisptr.frame_mode = _cdff_types.asn1Sccmode_png
        else:
            raise ValueError("Unknown frame_mode: %s" % frame_mode)

    frame_mode = property(_get_frame_mode, _set_frame_mode)


    def _get_frame_status(self):
        if <int> self.thisptr.frame_status == <int> _cdff_types.asn1Sccstatus_empty:
            return "status_empty"
        elif <int> self.thisptr.frame_status == <int> _cdff_types.asn1Sccstatus_valid:
            return "status_valid"
        else:
            return "status_invalid"

    def _set_frame_status(self,  str frame_status):
        if frame_status == "status_empty":
            self.thisptr.frame_status = _cdff_types.asn1Sccstatus_empty
        elif frame_status == "status_valid":
            self.thisptr.frame_status = _cdff_types.asn1Sccstatus_valid
        elif frame_status == "status_invalid":
            self.thisptr.frame_status = _cdff_types.asn1Sccstatus_invalid
        else:
            raise ValueError("Unknown frame_status: %s" % frame_status)

    frame_status = property(_get_frame_status, _set_frame_status)

    def array_reference(self, dtype=None):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.thisptr.datasize.height
        shape[1] = <np.npy_intp> self.thisptr.datasize.width
        shape[2] = <np.npy_intp> 3
        return np.PyArray_SimpleNewFromData(
            3, shape, np.NPY_UINT8, <void*> self.thisptr.image.arr)


cdef class Image_size_tReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def _get_width(self):
        return self.thisptr.width

    def _set_width(self, uint16_t width):
        self.thisptr.width = width

    width = property(_get_width, _set_width)

    def _get_height(self):
        return self.thisptr.height

    def _set_height(self, uint16_t height):
        self.thisptr.height = height

    height = property(_get_height, _set_height)


cdef class Image_imageReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        return self.thisptr.arr[i]

    def __setitem__(self, int i, unsigned char v):
        if i >= 24883200:
            warnings.warn("Maximum size of image is 24883200")
            return
        self.thisptr.arr[i] = v
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, int size):
        if size > 24883200:
            warnings.warn("Maximum size of image is 24883200")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount

    def fill(self, list data):
        cdef int data_len = len(data)
        self.resize(data_len)

        cdef int i
        for i in range(data_len):
            self.thisptr.arr[i] = data[i]


cdef class Image_attrib_tReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def _get_data(self):
        cdef bytes data = self.thisptr.data.arr
        return data.decode()

    def _set_data(self, str data):
        cdef string value = data.encode()
        memcpy(self.thisptr.data.arr, value.c_str(), len(data))
        self.thisptr.data.nCount = len(data)

    data = property(_get_data, _set_data)

    def _get_att_name(self):
        cdef bytes att_name = self.thisptr.att_name.arr
        return att_name.decode()

    def _set_att_name(self, str att_name):
        cdef string value = att_name.encode()
        memcpy(self.thisptr.att_name.arr, value.c_str(), len(att_name))
        self.thisptr.att_name.nCount = len(att_name)

    att_name = property(_get_att_name, _set_att_name)


cdef class Image_attributesReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        if i >= 5:
            raise KeyError("Maximum size of Image_attributes is 5")
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)
        cdef Image_attrib_tReference image_attrib = Image_attrib_tReference()
        image_attrib.thisptr = &self.thisptr.arr[i]
        return image_attrib

    def resize(self, int size):
        if size > 5:
            warnings.warn("Maximum size of Image_attributes is 5")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


class GpsSolution:
    def __init__(self):
        self.time = Time()
        self.latitude = 0.0
        self.longitude = 0.0
        self.position_type = "NO_SOLUTION"
        self.no_of_satellites = 0
        self.altitude = 0.0
        self.geoidal_separation = 0.0
        self.age_of_differential_corrections = 0.0
        self.deviation_latitude = 0.0
        self.deviation_longitude = 0.0
        self.deviation_altitude = 0.0

    def __str__(self):
        return str(
            "{type: GpsSolution, time=%s, latitude=%g, longitude=%g, "
            "position_type=%s, no_of_satellites=%d, altitude=%g, "
            "geoidal_separation=%g, age_of_differential_corrections=%g, "
            "deviation_latitude=%g, deviation_longitude=%g, "
            "deviation_altitude=%g}"
            % (self.time, self.latitude, self.longitude, self.position_type,
               self.no_of_satellites, self.altitude, self.geoidal_separation,
               self.age_of_differential_corrections, self.deviation_latitude,
               self.deviation_longitude, self.deviation_altitude))


cdef class Frame:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccFrame()
        self.delete_thisptr = True
        _cdff_types.asn1SccFrame_Initialize(self.thisptr)
        self.thisptr.msgVersion = _cdff_types.frame_Version
        self.thisptr.data.msgVersion = _cdff_types.array3D_Version

    def __str__(self):
        return str(
            "{type: Frame, msg_version: %d, metadata: %s, intrinsic: %s, "
            "extrinsic: %s, data: %s}"
            % (self.thisptr.msgVersion, self.metadata, self.intrinsic,
               self.extrinsic, self.data))

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    @property
    def metadata(self):
        cdef Frame_metadata_tReference metadata = Frame_metadata_tReference()
        metadata.thisptr = &self.thisptr.metadata
        return metadata

    @property
    def intrinsic(self):
        cdef Frame_intrinsic_tReference intrinsic = Frame_intrinsic_tReference()
        intrinsic.thisptr = &self.thisptr.intrinsic
        return intrinsic

    @property
    def extrinsic(self):
        cdef Frame_extrinsic_tReference extrinsic = Frame_extrinsic_tReference()
        extrinsic.thisptr = &self.thisptr.extrinsic
        return extrinsic

    @property
    def data(self):
        cdef Array3DReference data = Array3DReference()
        data.thisptr = &self.thisptr.data
        return data

    def array_reference(self, dtype=None):
        return self.data.array_reference()


cdef class FramePair:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccFramePair()
        self.delete_thisptr = True
        _cdff_types.asn1SccFramePair_Initialize(self.thisptr)

    def __str__(self):
        return ("{type: FramePair, msg_version: %d, baseline: %g, "
                "left: %s, right: %s}"
                % (self.thisptr.msgVersion, self.thisptr.baseline,
                   self.left, self.right))

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    def _get_baseline(self):
        return self.thisptr.baseline

    def _set_baseline(self, double baseline):
        self.thisptr.baseline = baseline

    baseline = property(_get_baseline, _set_baseline)

    def _set_left(self, Frame left):
        self.thisptr.left = deref(left.thisptr)

    def _get_left(self):
        cdef Frame left = Frame()
        del left.thisptr
        left.thisptr = &self.thisptr.left
        left.delete_thisptr = False
        return left

    left = property(_get_left, _set_left)

    def _set_right(self, Frame right):
        self.thisptr.right = deref(right.thisptr)

    def _get_right(self):
        cdef Frame right = Frame()
        del right.thisptr
        right.thisptr = &self.thisptr.right
        right.delete_thisptr = False
        return right

    right = property(_get_right, _set_right)


cdef class Frame_metadata_tReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        # TODO err_values, attributes
        return ("{msg_version: %d, time_stamp: %s, received_time: %s, "
                "pixel_model: %s, pixel_coeffs: %s, err_values: ..., "
                "attributes: ..., mode: %s, status: %s}"
                % (self.thisptr.msgVersion, self.time_stamp,
                   self.received_time, self.pixel_model, self.pixel_coeffs,
                   self.mode, self.status))

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    def _get_time_stamp(self):
        cdef Time time_stamp = Time()
        del time_stamp.thisptr
        time_stamp.thisptr = &self.thisptr.timeStamp
        time_stamp.delete_thisptr = False
        return time_stamp

    def _set_time_stamp(self, Time time_stamp):
        self.thisptr.timeStamp = deref(time_stamp.thisptr)

    time_stamp = property(_get_time_stamp, _set_time_stamp)

    def _get_received_time(self):
        cdef Time received_time = Time()
        del received_time.thisptr
        received_time.thisptr = &self.thisptr.receivedTime
        received_time.delete_thisptr = False
        return received_time

    def _set_received_time(self, Time received_time):
        self.thisptr.receivedTime = deref(received_time.thisptr)

    received_time = property(_get_received_time, _set_received_time)

    def _get_pixel_model(self):
        if <int> self.thisptr.pixelModel == <int> _cdff_types.asn1Sccpix_UNDEF:
            return "pix_UNDEF"
        elif <int> self.thisptr.pixelModel == <int> _cdff_types.asn1Sccpix_POLY:
            return "pix_POLY"
        elif <int> self.thisptr.pixelModel == <int> _cdff_types.asn1Sccpix_DISP:
            return "pix_DISP"
        else:
            raise ValueError("Unknown pixel model: %d" % <int> self.thisptr.pixelModel)

    def _set_pixel_model(self, str pixel_model):
        if pixel_model == "pix_UNDEF":
            self.thisptr.pixelModel = _cdff_types.asn1Sccpix_UNDEF
        elif pixel_model == "pix_POLY":
            self.thisptr.pixelModel = _cdff_types.asn1Sccpix_POLY
        elif pixel_model == "pix_DISP":
            self.thisptr.pixelModel = _cdff_types.asn1Sccpix_DISP
        else:
            raise ValueError("Unknown pixel model: %s" % pixel_model)

    pixel_model = property(_get_pixel_model, _set_pixel_model)

    @property
    def pixel_coeffs(self):
        cdef VectorXd pixel_coeffs = VectorXd()
        del pixel_coeffs.thisptr
        pixel_coeffs.delete_thisptr = False
        pixel_coeffs.thisptr = &self.thisptr.pixelCoeffs
        return pixel_coeffs

    @property
    def err_values(self):
        cdef Frame_metadata_t_errValuesReference err_values = \
            Frame_metadata_t_errValuesReference()
        err_values.thisptr = &self.thisptr.errValues
        return err_values

    @property
    def attributes(self):
        cdef Frame_metadata_t_attributesReference attributes = Frame_metadata_t_attributesReference()
        attributes.thisptr = &self.thisptr.attributes
        return attributes

    def _get_frame_mode(self):
        if <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_UNDEF:
            return "mode_UNDEF"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_GRAY:
            return "mode_GRAY"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_RGB:
            return "mode_RGB"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_RGBA:
            return "mode_RGBA"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_BGR:
            return "mode_BGR"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_BGRA:
            return "mode_BGRA"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_HSV:
            return "mode_HSV"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_HLS:
            return "mode_HLS"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_YUV:
            return "mode_YUV"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_UYVY:
            return "mode_UYVY"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_Lab:
            return "mode_Lab"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_Luv:
            return "mode_Luv"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_XYZ:
            return "mode_XYZ"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_YCrCb:
            return "mode_YCrCb"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_RGB32:
            return "mode_RGB32"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_Bayer_RGGB:
            return "mode_Bayer_RGGB"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_Bayer_GRBG:
            return "mode_Bayer_GRBG"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_Bayer_BGGR:
            return "mode_Bayer_BGGR"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_Bayer_GBRG:
            return "mode_Bayer_GBRG"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_PJPG:
            return "mode_PJPG"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_JPEG:
            return "mode_JPEG"
        elif <int> self.thisptr.mode == <int> _cdff_types.asn1Sccmode_PNG:
            return "mode_PNG"
        else:
            raise ValueError("Unknown frame mode: %d" % <int> self.thisptr.mode)

    def _set_frame_mode(self, str frame_mode):
        if frame_mode == "mode_UNDEF":
            self.thisptr.mode = _cdff_types.asn1Sccmode_UNDEF
        elif frame_mode == "mode_GRAY":
            self.thisptr.mode = _cdff_types.asn1Sccmode_GRAY
        elif frame_mode == "mode_RGB":
            self.thisptr.mode = _cdff_types.asn1Sccmode_RGB
        elif frame_mode == "mode_RGBA":
            self.thisptr.mode = _cdff_types.asn1Sccmode_RGBA
        elif frame_mode == "mode_BGR":
            self.thisptr.mode = _cdff_types.asn1Sccmode_BGR
        elif frame_mode == "mode_BGRA":
            self.thisptr.mode = _cdff_types.asn1Sccmode_BGRA
        elif frame_mode == "mode_HSV":
            self.thisptr.mode = _cdff_types.asn1Sccmode_HSV
        elif frame_mode == "mode_HLS":
            self.thisptr.mode = _cdff_types.asn1Sccmode_HLS
        elif frame_mode == "mode_YUV":
            self.thisptr.mode = _cdff_types.asn1Sccmode_YUV
        elif frame_mode == "mode_UYVY":
            self.thisptr.mode = _cdff_types.asn1Sccmode_UYVY
        elif frame_mode == "mode_Lab":
            self.thisptr.mode = _cdff_types.asn1Sccmode_Lab
        elif frame_mode == "mode_Luv":
            self.thisptr.mode = _cdff_types.asn1Sccmode_Luv
        elif frame_mode == "mode_XYZ":
            self.thisptr.mode = _cdff_types.asn1Sccmode_XYZ
        elif frame_mode == "mode_YCrCb":
            self.thisptr.mode = _cdff_types.asn1Sccmode_YCrCb
        elif frame_mode == "mode_RGB32":
            self.thisptr.mode = _cdff_types.asn1Sccmode_RGB32
        elif frame_mode == "mode_Bayer_RGGB":
            self.thisptr.mode = _cdff_types.asn1Sccmode_Bayer_RGGB
        elif frame_mode == "mode_Bayer_GRBG":
            self.thisptr.mode = _cdff_types.asn1Sccmode_Bayer_GRBG
        elif frame_mode == "mode_Bayer_BGGR":
            self.thisptr.mode = _cdff_types.asn1Sccmode_Bayer_BGGR
        elif frame_mode == "mode_Bayer_GBRG":
            self.thisptr.mode = _cdff_types.asn1Sccmode_Bayer_GBRG
        elif frame_mode == "mode_PJPG":
            self.thisptr.mode = _cdff_types.asn1Sccmode_PJPG
        elif frame_mode == "mode_JPEG":
            self.thisptr.mode = _cdff_types.asn1Sccmode_JPEG
        elif frame_mode == "mode_PNG":
            self.thisptr.mode = _cdff_types.asn1Sccmode_PNG
        else:
            raise ValueError("Unknown frame mode: %s" % frame_mode)

    mode = property(_get_frame_mode, _set_frame_mode)

    def _get_frame_status(self):
        if <int> self.thisptr.status == <int> _cdff_types.asn1Sccstatus_EMPTY:
            return "status_EMPTY"
        elif <int> self.thisptr.status == <int> _cdff_types.asn1Sccstatus_VALID:
            return "status_VALID"
        elif <int> self.thisptr.status == <int> _cdff_types.asn1Sccstatus_INVALID:
            return "status_INVALID"
        else:
            raise ValueError("Unknown frame status: %d" % <int> self.thisptr.status)

    def _set_frame_status(self, str frame_status):
        if frame_status == "status_EMPTY":
            self.thisptr.status = _cdff_types.asn1Sccstatus_EMPTY
        elif frame_status == "status_VALID":
            self.thisptr.status = _cdff_types.asn1Sccstatus_VALID
        elif frame_status == "status_INVALID":
            self.thisptr.status = _cdff_types.asn1Sccstatus_INVALID
        else:
            raise ValueError("Unknown frame status: %s" % frame_status)

    status = property(_get_frame_status, _set_frame_status)


cdef class Frame_metadata_t_attributesReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        if i >= 5:
            raise KeyError("Maximum size of attributes is 5")
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)
        cdef Frame_attrib_tReference attrib = Frame_attrib_tReference()
        attrib.thisptr = &self.thisptr.arr[i]
        return attrib

    def resize(self, int size):
        if size > 5:
            warnings.warn("Maximum size of attributes is 5")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class Frame_attrib_tReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def _get_data(self):
        cdef bytes data = self.thisptr.data.arr
        return data.decode()

    def _set_data(self, str data):
        cdef string value = data.encode()
        memcpy(self.thisptr.data.arr, value.c_str(), len(data))
        self.thisptr.data.nCount = len(data)

    data = property(_get_data, _set_data)

    def _get_name(self):
        cdef bytes name = self.thisptr.name.arr
        return name.decode()

    def _set_name(self, str name):
        cdef string value = name.encode()
        memcpy(self.thisptr.name.arr, value.c_str(), len(name))
        self.thisptr.name.nCount = len(name)

    name = property(_get_name, _set_name)


cdef class Frame_intrinsic_tReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return ("{msg_version: %d, sensor_id: %s, camera_matrix: %s, "
                "camera_model: %s, dist_coeffs: %s}"
                % (self.thisptr.msgVersion, self.sensor_id, self.camera_matrix,
                   self.camera_model, self.dist_coeffs))

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    def _get_sensor_id(self):
        cdef bytes sensorId = self.thisptr.sensorId.arr
        return sensorId.decode()

    def _set_sensor_id(self, str sensorId):
        cdef string value = sensorId.encode()
        memcpy(self.thisptr.sensorId.arr, value.c_str(), len(sensorId))
        self.thisptr.sensorId.nCount = len(sensorId)

    sensor_id = property(_get_sensor_id, _set_sensor_id)

    @property
    def camera_matrix(self):
        cdef Matrix3d camera_matrix = Matrix3d()
        del camera_matrix.thisptr
        camera_matrix.delete_thisptr = False
        camera_matrix.thisptr = &self.thisptr.cameraMatrix
        return camera_matrix

    def _get_camera_model(self):
        if <int> self.thisptr.cameraModel == <int> _cdff_types.asn1Scccam_UNDEF:
            return "cam_UNDEF"
        elif <int> self.thisptr.cameraModel == <int> _cdff_types.asn1Scccam_PINHOLE:
            return "cam_PINHOLE"
        elif <int> self.thisptr.cameraModel == <int> _cdff_types.asn1Scccam_FISHEYE:
            return "cam_FISHEYE"
        elif <int> self.thisptr.cameraModel == <int> _cdff_types.asn1Scccam_MAPS:
            return "cam_MAPS"
        else:
            raise ValueError("Unknown camera model: %d" % <int> self.thisptr.cameraModel)

    def _set_camera_model(self, str camera_model):
        if camera_model == "cam_UNDEF":
            self.thisptr.cameraModel = _cdff_types.asn1Scccam_UNDEF
        elif camera_model == "cam_PINHOLE":
            self.thisptr.cameraModel = _cdff_types.asn1Scccam_PINHOLE
        elif camera_model == "cam_FISHEYE":
            self.thisptr.cameraModel = _cdff_types.asn1Scccam_FISHEYE
        elif camera_model == "cam_MAPS":
            self.thisptr.cameraModel = _cdff_types.asn1Scccam_MAPS
        else:
            raise ValueError("Unknown camera model: %s" % camera_model)

    camera_model = property(_get_camera_model, _set_camera_model)

    @property
    def dist_coeffs(self):
        cdef VectorXd dist_coeffs = VectorXd()
        del dist_coeffs.thisptr
        dist_coeffs.delete_thisptr = False
        dist_coeffs.thisptr = &self.thisptr.distCoeffs
        return dist_coeffs


cdef class Frame_extrinsic_tReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return ("{msg_version: %d, has_fixed_transform: %s, "
                "pose_robot_frame_sensor_frame: %s, "
                "pose_fixed_frame_robot_frame: %s}"
                % (self.msg_version, self.has_fixed_transform,
                   self.pose_robot_frame_sensor_frame,
                   self.pose_fixed_frame_robot_frame))

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    def _get_has_fixed_transform(self):
        return self.thisptr.hasFixedTransform

    def _set_has_fixed_transform(self, bool has_fixed_transform):
        self.thisptr.hasFixedTransform = has_fixed_transform

    has_fixed_transform = property(
        _get_has_fixed_transform, _set_has_fixed_transform)

    @property
    def pose_robot_frame_sensor_frame(self):
        cdef TransformWithCovariance pose_robotFrame_sensorFrame = \
            TransformWithCovariance()
        del pose_robotFrame_sensorFrame.thisptr
        pose_robotFrame_sensorFrame.delete_thisptr = False
        pose_robotFrame_sensorFrame.thisptr = \
            &self.thisptr.pose_robotFrame_sensorFrame
        return pose_robotFrame_sensorFrame

    @property
    def pose_fixed_frame_robot_frame(self):
        cdef TransformWithCovariance pose_fixedFrame_robotFrame = \
            TransformWithCovariance()
        del pose_fixedFrame_robotFrame.thisptr
        pose_fixedFrame_robotFrame.delete_thisptr = False
        pose_fixedFrame_robotFrame.thisptr = \
            &self.thisptr.pose_fixedFrame_robotFrame
        return pose_fixedFrame_robotFrame


cdef class Array3DReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return ("{rows: %d, cols: %d, channels: %d, depth: %s, "
                "row_size: %d}"
                % (self.rows, self.cols, self.channels, self.depth,
                   self.row_size))

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    def _get_rows(self):
        return self.thisptr.rows

    def _set_rows(self, uint32_t rows):
        self.thisptr.rows = rows

    rows = property(_get_rows, _set_rows)

    def _get_cols(self):
        return self.thisptr.cols

    def _set_cols(self, uint32_t cols):
        self.thisptr.cols = cols

    cols = property(_get_cols, _set_cols)

    def _get_channels(self):
        return self.thisptr.channels

    def _set_channels(self, uint32_t channels):
        self.thisptr.channels = channels

    channels = property(_get_channels, _set_channels)

    def _get_depth(self):
        if <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_8U:
            return "depth_8U"
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_8S:
            return "depth_8S"
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_16U:
            return "depth_16U"
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_16S:
            return "depth_16S"
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_32S:
            return "depth_32S"
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_32F:
            return "depth_32F"
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_64F:
            return "depth_64F"
        else:
            raise ValueError("Unknown depth: %d" % <int> self.thisptr.depth)

    def _set_depth(self,  str depth):
        if depth == "depth_8U":
            self.thisptr.depth = _cdff_types.asn1Sccdepth_8U
        elif depth == "depth_8S":
            self.thisptr.depth = _cdff_types.asn1Sccdepth_8S
        elif depth == "depth_16U":
            self.thisptr.depth = _cdff_types.asn1Sccdepth_16U
        elif depth == "depth_16S":
            self.thisptr.depth = _cdff_types.asn1Sccdepth_16S
        elif depth == "depth_32S":
            self.thisptr.depth = _cdff_types.asn1Sccdepth_32S
        elif depth == "depth_32F":
            self.thisptr.depth = _cdff_types.asn1Sccdepth_32F
        elif depth == "depth_64F":
            self.thisptr.depth = _cdff_types.asn1Sccdepth_64F
        else:
            raise ValueError("Unknown depth: %s" % depth)

    depth = property(_get_depth, _set_depth)

    def _get_row_size(self):
        return self.thisptr.rowSize

    def _set_row_size(self, uint32_t row_size):
        self.thisptr.rowSize = row_size

    row_size = property(_get_row_size, _set_row_size)

    @property
    def data(self):
        cdef Array3D_dataReference data = Array3D_dataReference()
        data.thisptr = &self.thisptr.data
        return data

    def array_reference(self):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.thisptr.rows
        shape[1] = <np.npy_intp> self.thisptr.cols
        shape[2] = <np.npy_intp> self.thisptr.channels

        if <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_8U:
            array_type = np.NPY_UINT8
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_8S:
            array_type = np.NPY_INT8
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_16U:
            array_type = np.NPY_UINT16
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_16S:
            array_type = np.NPY_INT16
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_32S:
            array_type = np.NPY_INT32
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_32F:
            array_type = np.NPY_FLOAT32
        elif <int> self.thisptr.depth == <int> _cdff_types.asn1Sccdepth_64F:
            array_type = np.NPY_FLOAT64
        else:
            raise ValueError("Unknown depth: %d" % <int> self.thisptr.depth)

        return np.PyArray_SimpleNewFromData(
            3, shape, array_type, <void*> self.thisptr.data.arr)


cdef class Array3D_dataReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        return self.thisptr.arr[i]

    def __setitem__(self, int i, unsigned char v):
        if i >= 66355200:
            warnings.warn("Maximum size of Array3D is 66355200")
            return
        self.thisptr.arr[i] = v
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)

    def resize(self, int size):
        if size > 66355200:
            warnings.warn("Maximum size of Array3D is 66355200")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount

    def fill(self, list data):
        cdef int data_len = len(data)
        self.resize(data_len)

        cdef int i
        for i in range(data_len):
            self.thisptr.arr[i] = data[i]


cdef class Frame_metadata_t_errValuesReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        if i >= 3:
            warnings.warn("Maximum size of error values in frame metadata is 3")
            return
        cdef Frame_error_tReference entry = Frame_error_tReference()
        entry.thisptr = &self.thisptr.arr[i]
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)
        return entry

    def resize(self, int size):
        if size > 3:
            warnings.warn("Maximum size of error values in frame metadata is 3")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class Frame_error_tReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return "{type: %s, value: %s}" % (self.type, self.value)

    def _get_type(self):
        if <int> self.thisptr.type == <int> _cdff_types.asn1Sccerror_UNDEFINED:
            return "error_UNDEFINED"
        elif <int> self.thisptr.type == <int> _cdff_types.asn1Sccerror_DEAD:
            return "error_DEAD"
        elif <int> self.thisptr.type == <int> _cdff_types.asn1Sccerror_FILTERED:
            return "error_FILTERED"
        else:
            raise ValueError("Unknown error type: %d" % <int> self.thisptr.type)

    def _set_type(self, str type):
        if type == "error_UNDEFINED":
            self.thisptr.type = _cdff_types.asn1Sccerror_UNDEFINED
        elif type == "error_DEAD":
            self.thisptr.type = _cdff_types.asn1Sccerror_DEAD
        elif type == "error_FILTERED":
            self.thisptr.type = _cdff_types.asn1Sccerror_FILTERED
        else:
            raise ValueError("Unknown error type: %s" % type)

    type = property(_get_type, _set_type)

    def _get_value(self):
        return self.thisptr.value

    def _set_value(self, double value):
        self.thisptr.value = value

    value = property(_get_value, _set_value)


cdef class Map:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.asn1SccMap()
        self.delete_thisptr = True
        _cdff_types.asn1SccMap_Initialize(self.thisptr)
        self.thisptr.msgVersion = _cdff_types.map_Version
        self.thisptr.data.msgVersion = _cdff_types.array3D_Version

    def __str__(self):
        return ("{type: Map, metadata: %s, data: %s}"
                % (self.metadata, self.data))

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    @property
    def metadata(self):
        cdef Map_metadata_tReference metadata = Map_metadata_tReference()
        metadata.thisptr = &self.thisptr.metadata
        return metadata

    @property
    def data(self):
        cdef Array3DReference data = Array3DReference()
        data.thisptr = &self.thisptr.data
        return data

    def from_uper(self, list data):
        cdef unsigned char* uper_buffer = <unsigned char*> malloc(
            _cdff_types.asn1SccMap_REQUIRED_BYTES_FOR_ENCODING *
            sizeof(unsigned char))
        cdef _cdff_types.BitStream* b = new _cdff_types.BitStream()
        cdef int i
        _cdff_types.BitStream_Init(
            b, uper_buffer, _cdff_types.asn1SccMap_REQUIRED_BYTES_FOR_ENCODING)
        for i in range(_cdff_types.asn1SccMap_REQUIRED_BYTES_FOR_ENCODING):
            uper_buffer[i] = data[i]
            b.buf[i] = data[i]
        b.count = _cdff_types.asn1SccMap_REQUIRED_BYTES_FOR_ENCODING

        cdef int error_code
        cdef bool success
        success = _cdff_types.asn1SccMap_Decode(self.thisptr, b, &error_code)
        if not success:
            raise RuntimeError(
                "Conversion from uPER failed, error code: %d" % error_code)

        free(uper_buffer)
        del b


cdef class Map_metadata_tReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return ("{time_stamp: %s, type: %s, err_values: %s, scale: %g, "
                "pose_fixed_frame_map_frame: %s}"
                % (self.time_stamp, self.type, self.err_values, self.scale,
                   self.pose_fixed_frame_map_frame))

    def _get_msg_version(self):
        return self.thisptr.msgVersion

    def _set_msg_version(self, uint32_t msg_version):
        self.thisptr.msgVersion = msg_version

    msg_version = property(_get_msg_version, _set_msg_version)

    def _get_time_stamp(self):
        cdef Time time_stamp = Time()
        del time_stamp.thisptr
        time_stamp.thisptr = &self.thisptr.timeStamp
        time_stamp.delete_thisptr = False
        return time_stamp

    def _set_time_stamp(self, Time time_stamp):
        self.thisptr.timeStamp = deref(time_stamp.thisptr)

    time_stamp = property(_get_time_stamp, _set_time_stamp)

    def _get_type(self):
        if <int> self.thisptr.type == <int> _cdff_types.asn1Sccmap_UNDEF:
            return "map_UNDEF"
        elif <int> self.thisptr.type == <int> _cdff_types.asn1Sccmap_DEM:
            return "map_DEM"
        elif <int> self.thisptr.type == <int> _cdff_types.asn1Sccmap_NAV:
            return "map_NAV"
        else:
            raise ValueError("Unknown type: %d" % <int> self.thisptr.type)

    def _set_type(self,  str type):
        if type == "map_UNDEF":
            self.thisptr.type = _cdff_types.asn1Sccmap_UNDEF
        elif type == "map_DEM":
            self.thisptr.type = _cdff_types.asn1Sccmap_DEM
        elif type == "map_NAV":
            self.thisptr.type = _cdff_types.asn1Sccmap_NAV
        else:
            raise ValueError("Unknown type: %s" % type)

    type = property(_get_type, _set_type)

    @property
    def err_values(self):
        cdef Map_metadata_t_errValuesReference err_values = \
            Map_metadata_t_errValuesReference()
        err_values.thisptr = &self.thisptr.errValues
        return err_values

    def _get_scale(self):
        return self.thisptr.scale

    def _set_scale(self, double scale):
        self.thisptr.scale = scale

    scale = property(_get_scale, _set_scale)

    @property
    def pose_fixed_frame_map_frame(self):
        cdef TransformWithCovariance pose_fixedFrame_mapFrame = \
            TransformWithCovariance()
        del pose_fixedFrame_mapFrame.thisptr
        pose_fixedFrame_mapFrame.delete_thisptr = False
        pose_fixedFrame_mapFrame.thisptr = \
            &self.thisptr.pose_fixedFrame_mapFrame
        return pose_fixedFrame_mapFrame


cdef class Map_metadata_t_errValuesReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __str__(self):
        return "[%s]" % (", ".join([str(self[i]) for i in range(self.size())]))

    def __len__(self):
        return self.thisptr.nCount

    def __getitem__(self, int i):
        if i >= 5:
            warnings.warn("Maximum size of image is 5")
            return
        cdef Frame_error_tReference entry = Frame_error_tReference()
        entry.thisptr = &self.thisptr.arr[i]
        if self.thisptr.nCount <= <int> i:
            self.thisptr.nCount = <int> (i + 1)
        return entry

    def resize(self, int size):
        if size > 5:
            warnings.warn("Maximum size of image is 5")
            return
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount
