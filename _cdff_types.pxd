# distutils: language = c++
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.string cimport const_uchar


cdef extern from "Time.h":
    cdef cppclass asn1SccTime:
        asn1SccTime& assign "operator="(asn1SccTime&)

        int64_t microseconds
        int32_t usecPerSec


cdef extern from "Time.h":
    cdef enum asn1SccResolution:
        asn1Sccseconds
        asn1Sccmilliseconds
        asn1Sccmicroseconds


cdef extern from "Eigen.h":
    cdef cppclass asn1SccVector2d:
        asn1SccVector2d& assign "operator="(asn1SccVector2d&)

        int nCount
        double[2] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccVector3d:
        asn1SccVector3d& assign "operator="(asn1SccVector3d&)

        int nCount
        double[3] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccVector4d:
        asn1SccVector4d& assign "operator="(asn1SccVector4d&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccVector6d:
        asn1SccVector6d& assign "operator="(asn1SccVector6d&)

        int nCount
        double[6] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccVectorXd:
        asn1SccVectorXd& assign "operator="(asn1SccVectorXd&)

        int nCount
        double[100] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccMatrix2d_elm:
        asn1SccMatrix2d_elm& assign "operator="(asn1SccMatrix2d_elm&)

        int nCount
        double[2] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccMatrix2d:
        asn1SccMatrix2d& assign "operator="(asn1SccMatrix2d&)

        int nCount
        asn1SccMatrix2d_elm[2] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccMatrix3d_elm:
        asn1SccMatrix3d_elm& assign "operator="(asn1SccMatrix3d_elm&)

        int nCount
        double[3] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccMatrix3d:
        asn1SccMatrix3d& assign "operator="(asn1SccMatrix3d&)

        int nCount
        asn1SccMatrix3d_elm[3] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccMatrix4d_elm:
        asn1SccMatrix4d_elm& assign "operator="(asn1SccMatrix4d_elm&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccMatrix4d:
        asn1SccMatrix4d& assign "operator="(asn1SccMatrix4d&)

        int nCount
        asn1SccMatrix4d_elm[4] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccQuaterniond:
        asn1SccQuaterniond& assign "operator="(asn1SccQuaterniond&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccAngleAxisd:
        asn1SccAngleAxisd& assign "operator="(asn1SccAngleAxisd&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccTransform3d_elm:
        asn1SccTransform3d_elm& assign "operator="(asn1SccTransform3d_elm&)

        int nCount
        double[4] arr


cdef extern from "Eigen.h":
    cdef cppclass asn1SccTransform3d:
        asn1SccTransform3d& assign "operator="(asn1SccTransform3d&)

        int nCount
        asn1SccTransform3d_elm[4] arr

# TODO Matrix6d MatrixXd Isometry3d Affine3d


cdef extern from "Pointcloud.h":
    cdef uint32_t maxPointcloudSize

    cdef cppclass asn1SccPointcloud_colors:
        asn1SccPointcloud_colors& assign "operator="(asn1SccPointcloud_colors&)

        int nCount
        asn1SccVector4d[300000] arr

    cdef cppclass asn1SccPointcloud_points:
        asn1SccPointcloud_points& assign "operator="(asn1SccPointcloud_points&)

        int nCount
        asn1SccVector3d[300000] arr

    cdef cppclass asn1SccPointcloud:
        asn1SccPointcloud& assign "operator="(asn1SccPointcloud&)

        asn1SccTime ref_time
        asn1SccPointcloud_points points
        asn1SccPointcloud_colors colors


cdef extern from "LaserScan.h":
    cdef uint32_t maxLaserScanSize

    cdef cppclass asn1SccLaserScan_ranges:
        asn1SccLaserScan_ranges& assign "operator="(asn1SccLaserScan_ranges&)

        int nCount
        int32_t[60] arr

    cdef cppclass asn1SccLaserScan_remission:
        asn1SccLaserScan_remission& assign "operator="(asn1SccLaserScan_remission&)

        int nCount
        float[60] arr

    cdef cppclass asn1SccLaserScan:
        asn1SccLaserScan& assign "operator="(asn1SccLaserScan&)

        asn1SccTime ref_time;
        double start_angle
        double angular_resolution
        double speed
        asn1SccLaserScan_ranges ranges
        uint32_t minRange
        uint32_t maxRange
        asn1SccLaserScan_remission remission


cdef extern from "taste-extended.h":
    cdef cppclass asn1SccT_String:
        int nCount
        const_uchar[256] arr


cdef extern from "RigidBodyState.h":
    cdef cppclass asn1SccRigidBodyState:
            asn1SccTime timestamp
            asn1SccT_String sourceFrame
            asn1SccT_String targetFrame
            asn1SccVector3d pos
            asn1SccMatrix3d cov_position
            asn1SccQuaterniond orient
            asn1SccMatrix3d cov_orientation
            asn1SccVector3d velocity
            asn1SccMatrix3d cov_velocity
            asn1SccVector3d angular_velocity
            asn1SccMatrix3d cov_angular_velocity


cdef extern from "JointState.h":
    cdef cppclass asn1SccJointState:
        asn1SccJointState& assign "operator="(asn1SccJointState&)

        double position
        float speed
        float effort
        float raw
        float acceleration


cdef extern from "Joints.h":
    cdef cppclass asn1SccJoints_names:
        int nCount
        asn1SccT_String[30] arr

    cdef cppclass asn1SccJoints_elements:
        int nCount
        asn1SccJointState[30] arr

    cdef cppclass asn1SccJoints:
        asn1SccJoints& assign "operator="(asn1SccJoints&)

        asn1SccTime timestamp
        asn1SccJoints_names names
        asn1SccJoints_elements elements


cdef extern from "IMUSensors.h":
    cdef cppclass asn1SccIMUSensors:
        asn1SccTime timestamp
        asn1SccVector3d acc
        asn1SccVector3d gyro
        asn1SccVector3d mag
