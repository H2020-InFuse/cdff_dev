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


cdef class Matrix3d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.Matrix3d()
        self.thisptr.nCount = 3
        cdef int i
        for i in range(self.thisptr.nCount):
            self.thisptr.arr[i].nCount = self.thisptr.nCount
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("{type: Matrix3d, data=[?]}") # TODO print content

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

    def assign(self, Matrix3d other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=2] array = np.empty((3, 3))
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


cdef class Quaterniond:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.Quaterniond()
        self.thisptr.nCount = 4
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.nCount

    def __str__(self):
        return str("{type: Quaterniond, data=[%.2f, %.2f, %.2f, %.2f]}"
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


cdef class Vector3dVectorReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __getitem__(self, int i):
        cdef Vector3d v = Vector3d()
        del v.thisptr
        v.delete_thisptr = False
        v.thisptr = &(self.thisptr.arr[i])
        return v

    def resize(self, int size):
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class Vector4dVectorReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __getitem__(self, int i):
        cdef Vector4d v = Vector4d()
        del v.thisptr
        v.delete_thisptr = False
        v.thisptr = &(self.thisptr.arr[i])
        return v

    def resize(self, int size):
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class Pointcloud:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.Pointcloud()
        self.thisptr.points.nCount = 0
        self.thisptr.colors.nCount = 0
        self.delete_thisptr = True

    def __len__(self):
        return self.thisptr.points.nCount

    def __str__(self):
        return str("{type: Pointcloud, data=[?]}")  # TODO

    def assign(self, Pointcloud other):
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

    @property
    def points(self):
        cdef Vector3dVectorReference points = Vector3dVectorReference()
        points.thisptr = &self.thisptr.points
        return points

    @property
    def colors(self):
        cdef Vector4dVectorReference colors = Vector4dVectorReference()
        colors.thisptr = &self.thisptr.colors
        return colors


cdef class LaserScan:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.LaserScan()
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
        return str("{type: LaserScan, data=[?]}")  # TODO

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

    def __getitem__(self, int i):
        return self.thisptr.arr[i]

    def __setitem__(self, int i, float v):
        self.thisptr.arr[i] = v

    def resize(self, int size):
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class LaserScan_rangesReference:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        pass

    def __getitem__(self, int i):
        return self.thisptr.arr[i]

    def __setitem__(self, int i, uint32_t v):
        self.thisptr.arr[i] = v

    def resize(self, int size):
        self.thisptr.nCount = size

    def size(self):
        return self.thisptr.nCount


cdef class RigidBodyState:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_types.RigidBodyState()
        self.delete_thisptr = True

    def __str__(self):
        return str("{type: RigidBodyStateg, %s, sourceFrame=%s, targetFrame=%s, ...}"
                   % (self.timestamp, self.source_frame, self.target_frame))

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
        self.thisptr.sourceFrame.arr = value.c_str()
        self.thisptr.sourceFrame.nCount = len(sourceFrame)

    source_frame = property(_get_source_frame, _set_source_frame)

    def _get_target_frame(self):
        cdef bytes target_frame = self.thisptr.targetFrame.arr
        return target_frame.decode()

    def _set_target_frame(self, str targetFrame):
        cdef string value = targetFrame.encode()
        self.thisptr.targetFrame.arr = value.c_str()
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