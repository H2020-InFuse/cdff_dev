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


cdef extern from "Eigen.h":
    cdef cppclass Vector2d:
        Vector2d& assign "operator="(Vector2d&)

        int nCount
        double[2] arr


cdef extern from "Eigen.h":
    cdef cppclass Vector3d:
        Vector3d& assign "operator="(Vector3d&)

        int nCount
        double[3] arr


cdef extern from "Eigen.h":
    cdef cppclass Vector4d:
        Vector4d& assign "operator="(Vector4d&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass Vector6d:
        Vector6d& assign "operator="(Vector6d&)

        int nCount
        double[6] arr


cdef extern from "Eigen.h":
    cdef cppclass VectorXd:
        VectorXd& assign "operator="(VectorXd&)

        int nCount
        double[100] arr


cdef extern from "Eigen.h":
    cdef cppclass Matrix2d_elm:
        Matrix2d_elm& assign "operator="(Matrix2d_elm&)

        int nCount
        double[2] arr


cdef extern from "Eigen.h":
    cdef cppclass Matrix2d:
        Matrix2d& assign "operator="(Matrix2d&)

        int nCount
        Matrix2d_elm[2] arr


cdef extern from "Eigen.h":
    cdef cppclass Matrix3d_elm:
        Matrix3d_elm& assign "operator="(Matrix3d_elm&)

        int nCount
        double[3] arr


cdef extern from "Eigen.h":
    cdef cppclass Matrix3d:
        Matrix3d& assign "operator="(Matrix3d&)

        int nCount
        Matrix3d_elm[3] arr


cdef extern from "Eigen.h":
    cdef cppclass Matrix4d_elm:
        Matrix4d_elm& assign "operator="(Matrix4d_elm&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass Matrix4d:
        Matrix4d& assign "operator="(Matrix4d&)

        int nCount
        Matrix4d_elm[4] arr


cdef extern from "Eigen.h":
    cdef cppclass Quaterniond:
        Quaterniond& assign "operator="(Quaterniond&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass AngleAxisd:
        AngleAxisd& assign "operator="(AngleAxisd&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass Transform3d_elm:
        Transform3d_elm& assign "operator="(Transform3d_elm&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass Transform3d:
        Transform3d& assign "operator="(Transform3d&)

        int nCount
        Transform3d_elm[4] arr

# TODO Matrix6d MatrixXd Isometry3d Affine3d


cdef extern from "Pointcloud.h":
    cdef uint32_t maxPointcloudSize

    cdef cppclass Pointcloud_colors:
        Pointcloud_colors& assign "operator="(Pointcloud_colors&)

        int nCount
        Vector4d[30000] arr

    cdef cppclass Pointcloud_points:
        Pointcloud_points& assign "operator="(Pointcloud_points&)

        int nCount
        Vector3d[30000] arr

    cdef cppclass Pointcloud:
        Pointcloud& assign "operator="(Pointcloud&)

        Time ref_time
        Pointcloud_points points
        Pointcloud_colors colors

cdef extern from "LaserScan.h":
    cdef uint32_t maxLaserScanSize

    cdef cppclass LaserScan_ranges:
        LaserScan_ranges& assign "operator="(LaserScan_ranges&)

        int nCount
        int32_t[30000] arr

    cdef cppclass LaserScan_remission:
        LaserScan_remission& assign "operator="(LaserScan_remission&)

        int nCount
        float[30000] arr

    cdef cppclass LaserScan:
        LaserScan& assign "operator="(LaserScan&)

        Time ref_time;
        double start_angle
        double angular_resolution
        double speed
        LaserScan_ranges ranges
        uint32_t minRange
        uint32_t maxRange
        LaserScan_remission remission
