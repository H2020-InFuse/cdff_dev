# distutils: language = c++
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t


cdef extern from "Time.h":
    cdef cppclass Time:
        Time& assign "operator="(Time&)

        int64_t microseconds
        int32_t usecPerSec


cdef extern from "Time.h":
    cdef enum Resolution:
        seconds
        milliseconds
        microseconds
