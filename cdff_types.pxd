# distutils: language = c++
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
cimport _cdff_types


cdef class Time:
    cdef _cdff_types.asn1SccTime* thisptr
    cdef bool delete_thisptr


cdef class Vector2d:
    cdef _cdff_types.asn1SccVector2d* thisptr
    cdef bool delete_thisptr


cdef class Vector3d:
    cdef _cdff_types.asn1SccVector3d* thisptr
    cdef bool delete_thisptr


cdef class Vector4d:
    cdef _cdff_types.asn1SccVector4d* thisptr
    cdef bool delete_thisptr


cdef class Vector6d:
    cdef _cdff_types.asn1SccVector6d* thisptr
    cdef bool delete_thisptr


cdef class VectorXd:
    cdef _cdff_types.asn1SccVectorXd* thisptr
    cdef bool delete_thisptr


cdef class Matrix2d:
    cdef _cdff_types.asn1SccMatrix2d* thisptr
    cdef bool delete_thisptr


cdef class Matrix3d:
    cdef _cdff_types.asn1SccMatrix3d* thisptr
    cdef bool delete_thisptr


cdef class Quaterniond:
    cdef _cdff_types.asn1SccQuaterniond* thisptr
    cdef bool delete_thisptr


cdef class AngleAxisd:
    cdef _cdff_types.asn1SccAngleAxisd* thisptr
    cdef bool delete_thisptr


cdef class Transform3d:
    cdef _cdff_types.asn1SccTransform3d* thisptr
    cdef bool delete_thisptr


cdef class Pointcloud_points:
    cdef _cdff_types.asn1SccPointcloud_points* thisptr
    cdef bool delete_thisptr


cdef class Pointcloud_colors:
    cdef _cdff_types.asn1SccPointcloud_colors* thisptr
    cdef bool delete_thisptr


cdef class Pointcloud:
    cdef _cdff_types.asn1SccPointcloud* thisptr
    cdef bool delete_thisptr


cdef class LaserScan:
    cdef _cdff_types.asn1SccLaserScan* thisptr
    cdef bool delete_thisptr


cdef class LaserScan_remissionReference:
    cdef _cdff_types.asn1SccLaserScan_remission* thisptr


cdef class LaserScan_rangesReference:
    cdef _cdff_types.asn1SccLaserScan_ranges* thisptr


cdef class T_String:
    cdef _cdff_types.asn1SccT_String* thisptr


cdef class RigidBodyState:
    cdef _cdff_types.asn1SccRigidBodyState* thisptr
    cdef bool delete_thisptr


cdef class JointState:
    cdef _cdff_types.asn1SccJointState* thisptr
    cdef bool delete_thisptr


cdef class Joints:
    cdef _cdff_types.asn1SccJoints* thisptr
    cdef bool delete_thisptr


cdef class Joints_namesReference:
    cdef _cdff_types.asn1SccJoints_names* thisptr


cdef class Joints_elementsReference:
    cdef _cdff_types.asn1SccJoints_elements* thisptr


cdef class IMUSensors:
    cdef _cdff_types.asn1SccIMUSensors* thisptr
    cdef bool delete_thisptr


cdef class DepthMap:
    cdef _cdff_types.asn1SccDepthMap* thisptr
    cdef bool delete_thisptr


cdef class DepthMap_timestampsReference:
    cdef _cdff_types.asn1SccDepthMap_timestamps* thisptr


cdef class DepthMap_vertical_intervalReference:
    cdef _cdff_types.asn1SccDepthMap_vertical_interval* thisptr


cdef class DepthMap_horizontal_intervalReference:
    cdef _cdff_types.asn1SccDepthMap_horizontal_interval* thisptr


cdef class DepthMap_distancesReference:
    cdef _cdff_types.asn1SccDepthMap_distances* thisptr


cdef class DepthMap_remissionsReference:
    cdef _cdff_types.asn1SccDepthMap_remissions* thisptr


cdef class Frame:
    cdef _cdff_types.asn1SccFrame* thisptr
    cdef bool delete_thisptr


cdef class Frame_size_tReference:
    cdef _cdff_types.asn1SccFrame_size_t* thisptr


cdef class Frame_imageReference:
    cdef _cdff_types.asn1SccFrame_image* thisptr


cdef class Frame_attrib_tReference:
    cdef _cdff_types.asn1SccFrame_attrib_t* thisptr


cdef class Frame_attributesReference:
    cdef _cdff_types.asn1SccFrame_attributes* thisptr
