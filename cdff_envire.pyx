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