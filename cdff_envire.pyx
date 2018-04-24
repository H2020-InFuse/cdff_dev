# distutils: language = c++
from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t
cimport numpy as np
import numpy as np
cimport cdff_envire


np.import_array()  # must be here because we use the NumPy C API


cdef class Time:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_envire.Time()
        self.delete_thisptr = True

    def __str__(self):
        cdef bytes text = self.thisptr.toString(
            _cdff_envire.Resolution.Microseconds, "%Y%m%d-%H:%M:%S")
        return str("<time=%s>" % text.decode())

    def assign(self, Time other):
        self.thisptr.assign(deref(other.thisptr))

    def _get_microseconds(self):
        return self.thisptr.microseconds

    def _set_microseconds(self, int64_t microseconds):
        self.thisptr.microseconds = microseconds

    microseconds = property(_get_microseconds, _set_microseconds)

    def is_null(self):
        return self.thisptr.isNull()

    @staticmethod
    def now():
        cdef Time time = Time()
        time.thisptr[0] = _cdff_envire.now()
        return time

    def __richcmp__(Time self, Time other, int op):
        if op == 0:
            return deref(self.thisptr) < deref(other.thisptr)
        elif op == 1:
            return deref(self.thisptr) <= deref(other.thisptr)
        elif op == 2:
            return deref(self.thisptr) == deref(other.thisptr)
        elif op == 3:
            return deref(self.thisptr) != deref(other.thisptr)
        elif op == 4:
            return deref(self.thisptr) > deref(other.thisptr)
        elif op == 5:
            return deref(self.thisptr) >= deref(other.thisptr)
        else:
            raise ValueError("Unknown comparison operation %d" % op)

    def __add__(Time self, Time other):
        cdef Time time = Time()
        time.thisptr[0] = deref(self.thisptr) + deref(other.thisptr)
        return time

    def __iadd__(Time self, Time other):
        self.thisptr[0] = deref(self.thisptr) + deref(other.thisptr)
        return self

    def __sub__(Time self, Time other):
        cdef Time time = Time()
        time.thisptr[0] = deref(self.thisptr) - deref(other.thisptr)
        return time

    def __isub__(Time self, Time other):
        self.thisptr[0] = deref(self.thisptr) - deref(other.thisptr)
        return self

    def __floordiv__(Time self, int divider):
        cdef Time time = Time()
        time.thisptr[0] = deref(self.thisptr) / divider
        return time

    def __ifloordiv__(Time self, int divider):
        self.thisptr[0] = deref(self.thisptr) / divider
        return self

    def __mul__(Time self, double factor):
        cdef Time time = Time()
        time.thisptr[0] = deref(self.thisptr) * factor
        return time

    def __imul__(Time self, double factor):
        self.thisptr[0] = deref(self.thisptr) * factor
        return self


cdef class Vector3d:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self, double x=0.0, double y=0.0, double z=0.0):
        self.thisptr = new _cdff_envire.Vector3d(x, y, z)
        self.delete_thisptr = True

    def __str__(self):
        return str("[%.2f, %.2f, %.2f]" % (
                   self.thisptr.get(0), self.thisptr.get(1),
                   self.thisptr.get(2)))

    def assign(self, Vector3d other):
        self.thisptr.assign(deref(other.thisptr))

    def __array__(self, dtype=None):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 3
        return np.PyArray_SimpleNewFromData(
            1, shape, np.NPY_DOUBLE, <void*> self.thisptr.data())

    def __getitem__(self, int i):
        if i < 0 or i > 2:
            raise KeyError("index must be 0, 1 or 2 but was %d" % i)
        return self.thisptr.data()[i]

    def __setitem__(self, int i, double v):
        if i < 0 or i > 2:
            raise KeyError("index must be 0, 1 or 2 but was %d" % i)
        self.thisptr.data()[i] = v

    def _get_x(self):
        return self.thisptr.data()[0]

    def _set_x(self, double x):
        self.thisptr.data()[0] = x

    x = property(_get_x, _set_x)

    def _get_y(self):
        return self.thisptr.data()[1]

    def _set_y(self, double y):
        self.thisptr.data()[1] = y

    y = property(_get_y, _set_y)

    def _get_z(self):
        return self.thisptr.data()[2]

    def _set_z(self, double z):
        self.thisptr.data()[2] = z

    z = property(_get_z, _set_z)

    def norm(self):
        return self.thisptr.norm()

    def squared_norm(self):
        return self.thisptr.squaredNorm()

    def toarray(self):
        cdef np.ndarray[double, ndim=1] array = np.empty(3)
        cdef int i
        for i in range(3):
            array[i] = self.thisptr.data()[i]
        return array

    def fromarray(self, np.ndarray[double, ndim=1] array):
        cdef int i
        for i in range(3):
            self.thisptr.data()[i] = array[i]

    def __richcmp__(Vector3d self, Vector3d other, int op):
        if op == 0:
            raise NotImplementedError("<")
        elif op == 1:
            raise NotImplementedError("<=")
        elif op == 2:
            return deref(self.thisptr) == deref(other.thisptr)
        elif op == 3:
            return deref(self.thisptr) != deref(other.thisptr)
        elif op == 4:
            raise NotImplementedError(">")
        elif op == 5:
            raise NotImplementedError(">=")
        else:
            raise ValueError("Unknown comparison operation %d" % op)


cdef class Quaterniond:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self, double w=1.0, double x=0.0, double y=0.0, double z=0.0):
        self.thisptr = new _cdff_envire.Quaterniond(w, x, y, z)
        self.delete_thisptr = True

    def __str__(self):
        return str("[im=%.2f, real=(%.2f, %.2f, %.2f)]" % (
            self.thisptr.w(), self.thisptr.x(), self.thisptr.y(),
            self.thisptr.z()))

    def assign(self, Quaterniond other):
        self.thisptr.assign(deref(other.thisptr))

    def toarray(self):
        cdef np.ndarray[double, ndim=1] array = np.array([
            self.thisptr.w(), self.thisptr.x(), self.thisptr.y(),
            self.thisptr.z()])
        return array

    def fromarray(self, np.ndarray[double, ndim=1] array):
        self.thisptr[0] = _cdff_envire.Quaterniond(
            array[0], array[1], array[2], array[3])


cdef class TransformWithCovariance:
    def __cinit__(self):
        self.thisptr = NULL
        self.delete_thisptr = False

    def __dealloc__(self):
        if self.thisptr != NULL and self.delete_thisptr:
            del self.thisptr

    def __init__(self):
        self.thisptr = new _cdff_envire.TransformWithCovariance()
        self.delete_thisptr = True

    def __str__(self):
        return str("(translation=%s, orientation=%s)" % (self.translation,
                                                         self.orientation))

    def assign(self, TransformWithCovariance other):
        self.thisptr.assign(deref(other.thisptr))

    def _get_translation(self):
        cdef Vector3d translation = Vector3d()
        del translation.thisptr
        translation.delete_thisptr = False
        translation.thisptr = &self.thisptr.translation
        return translation

    def _set_translation(self, Vector3d translation):
        self.thisptr.translation = deref(translation.thisptr)

    translation = property(_get_translation, _set_translation)

    def _get_orientation(self):
        cdef Quaterniond orientation = Quaterniond()
        del orientation.thisptr
        orientation.delete_thisptr = False
        orientation.thisptr = &self.thisptr.orientation
        return orientation

    def _set_orientation(self, Quaterniond orientation):
        self.thisptr.orientation = deref(orientation.thisptr)

    orientation = property(_get_orientation, _set_orientation)