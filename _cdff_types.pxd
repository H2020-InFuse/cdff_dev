# distutils: language = c++
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.string cimport const_uchar
from libcpp cimport bool


cdef extern from "Types/C/Time.h":
    cdef cppclass asn1SccTime:
        asn1SccTime& assign "operator="(asn1SccTime&)

        int64_t microseconds
        int32_t usecPerSec


cdef extern from "Types/C/Time.h":
    cdef enum asn1SccResolution:
        asn1Sccseconds
        asn1Sccmilliseconds
        asn1Sccmicroseconds


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccVector2d:
        asn1SccVector2d& assign "operator="(asn1SccVector2d&)

        double[2] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccVector3d:
        asn1SccVector3d& assign "operator="(asn1SccVector3d&)

        double[3] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccVector4d:
        asn1SccVector4d& assign "operator="(asn1SccVector4d&)

        double[4] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccVector6d:
        asn1SccVector6d& assign "operator="(asn1SccVector6d&)

        double[6] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccVectorXd:
        asn1SccVectorXd& assign "operator="(asn1SccVectorXd&)

        int nCount
        double[100] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccMatrix2d_elm:
        asn1SccMatrix2d_elm& assign "operator="(asn1SccMatrix2d_elm&)

        double[2] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccMatrix2d:
        asn1SccMatrix2d& assign "operator="(asn1SccMatrix2d&)

        asn1SccMatrix2d_elm[2] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccMatrix3d_elm:
        asn1SccMatrix3d_elm& assign "operator="(asn1SccMatrix3d_elm&)

        double[3] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccMatrix3d:
        asn1SccMatrix3d& assign "operator="(asn1SccMatrix3d&)

        asn1SccMatrix3d_elm[3] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccMatrix4d_elm:
        asn1SccMatrix4d_elm& assign "operator="(asn1SccMatrix4d_elm&)

        double[4] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccMatrix4d:
        asn1SccMatrix4d& assign "operator="(asn1SccMatrix4d&)

        asn1SccMatrix4d_elm[4] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccMatrix6d_elm:
        asn1SccMatrix6d_elm& assign "operator="(asn1SccMatrix6d_elm&)

        double[6] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccMatrix6d:
        asn1SccMatrix6d& assign "operator="(asn1SccMatrix6d&)

        asn1SccMatrix6d_elm[6] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccQuaterniond:
        asn1SccQuaterniond& assign "operator="(asn1SccQuaterniond&)

        double[4] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccAngleAxisd:
        asn1SccAngleAxisd& assign "operator="(asn1SccAngleAxisd&)

        double[4] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccTransform3d_elm:
        asn1SccTransform3d_elm& assign "operator="(asn1SccTransform3d_elm&)

        double[4] arr


cdef extern from "Types/C/Eigen.h":
    cdef cppclass asn1SccTransform3d:
        asn1SccTransform3d& assign "operator="(asn1SccTransform3d&)

        asn1SccTransform3d_elm[4] arr


# TODO MatrixXd Isometry3d Affine3d


cdef extern from "Types/C/Pose.h":
    cdef cppclass asn1SccPose:
        asn1SccVector3d pos
        asn1SccQuaterniond orient


cdef extern from "Types/C/TransformWithCovariance.h":
    cdef cppclass asn1SccTransformWithCovariance_Data:
        asn1SccVector3d translation
        asn1SccQuaterniond orientation
        asn1SccMatrix6d cov

    cdef cppclass asn1SccTransformWithCovariance_Metadata_dataEstimated:
        bool[7] arr

    cdef cppclass asn1SccTransformWithCovariance_Metadata:
        uint32_t msgVersion
        asn1SccT_String producerId
        asn1SccTransformWithCovariance_Metadata_dataEstimated dataEstimated
        asn1SccT_String parentFrameId
        asn1SccTime parentTime
        asn1SccT_String childFrameId
        asn1SccTime childTime

    cdef cppclass asn1SccTransformWithCovariance:
        asn1SccTransformWithCovariance_Metadata metadata
        asn1SccTransformWithCovariance_Data data

    bool asn1SccTransformWithCovariance_Decode(asn1SccTransformWithCovariance* pVal, BitStream* pBitStrm, int* pErrCode)
    int asn1SccTransformWithCovariance_REQUIRED_BYTES_FOR_ENCODING


cdef extern from "Types/C/Pointcloud.h":
    cdef cppclass asn1SccPointCloud_Data_points:
        int nCount
        asn1SccVector3d[400000] arr

    cdef cppclass asn1SccPointCloud_Data_colors:
        int nCount
        asn1SccVector3d[400000] arr

    cdef cppclass asn1SccPointCloud_Data_intensity:
        int nCount
        int32_t[400000] arr

    cdef cppclass asn1SccPointCloud_Data:
        asn1SccPointCloud_Data_points points
        asn1SccPointCloud_Data_colors colors
        asn1SccPointCloud_Data_intensity intensity

    cdef cppclass asn1SccPointCloud_Metadata:
        uint32_t msgVersion
        asn1SccT_String sensorId
        asn1SccT_String frameId
        asn1SccTime timeStamp
        uint32_t height
        uint32_t width
        bool isRegistered
        bool isOrdered
        bool hasFixedTransform
        asn1SccTransformWithCovariance pose_robotFrame_sensorFrame
        asn1SccTransformWithCovariance pose_fixedFrame_robotFrame

    cdef cppclass asn1SccPointcloud:
        asn1SccPointCloud_Metadata metadata
        asn1SccPointCloud_Data data

    cdef uint32_t pointCloud_Version
    cdef void asn1SccPointcloud_Initialize(asn1SccPointcloud*)

    bool asn1SccPointcloud_Decode(asn1SccPointcloud* pVal, BitStream* pBitStrm, int* pErrCode)
    int asn1SccPointcloud_REQUIRED_BYTES_FOR_ENCODING


cdef extern from "Types/C/LaserScan.h":
    cdef uint32_t maxLaserScanSize

    cdef cppclass asn1SccLaserScan_ranges:
        asn1SccLaserScan_ranges& assign "operator="(asn1SccLaserScan_ranges&)

        int nCount
        int32_t[2000] arr

    cdef cppclass asn1SccLaserScan_remission:
        asn1SccLaserScan_remission& assign "operator="(asn1SccLaserScan_remission&)

        int nCount
        float[2000] arr

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


cdef extern from "Types/C/taste-extended.h":
    cdef cppclass asn1SccT_String:
        int nCount
        const_uchar[256] arr


cdef extern from "Types/C/RigidBodyState.h":
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


cdef extern from "Types/C/JointState.h":
    cdef cppclass asn1SccJointState:
        asn1SccJointState& assign "operator="(asn1SccJointState&)

        double position
        float speed
        float effort
        float raw
        float acceleration


cdef extern from "Types/C/Joints.h":
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


cdef extern from "Types/C/IMUSensors.h":
    cdef cppclass asn1SccIMUSensors:
        asn1SccTime timestamp
        asn1SccVector3d acc
        asn1SccVector3d gyro
        asn1SccVector3d mag


cdef extern from "Types/C/DepthMap.h":
    cdef enum asn1SccUNIT_AXIS:
        asn1Sccunit_x
        asn1Sccunit_y
        asn1Sccunit_z


cdef extern from "Types/C/DepthMap.h":
    cdef enum asn1SccDEPTH_MEASUREMENT_STATE:
        asn1Sccvalid_measurement
        asn1SccDEPTH_MEASUREMENT_STATE_too_far
        asn1SccDEPTH_MEASUREMENT_STATE_too_near
        asn1SccDEPTH_MEASUREMENT_STATE_measurement_error


cdef extern from "Types/C/DepthMap.h":
    cdef cppclass asn1SccPROJECTION_TYPE:
        pass


cdef extern from "Types/C/DepthMap.h" namespace "asn1SccPROJECTION_TYPE":
    cdef asn1SccPROJECTION_TYPE asn1Sccpolar
    cdef asn1SccPROJECTION_TYPE asn1Sccplanar


cdef extern from "Types/C/DepthMap.h":
    cdef cppclass asn1SccDepthMap_horizontal_interval:
        int nCount
        double arr[30000]
    cdef cppclass asn1SccDepthMap_vertical_interval:
        int nCount
        double arr[30000]
    cdef cppclass asn1SccDepthMap_remissions:
        int nCount
        double arr[30000]
    cdef cppclass asn1SccDepthMap_distances:
        int nCount
        double arr[30000]
    cdef cppclass asn1SccDepthMap_timestamps:
        int nCount
        asn1SccTime arr[30000]
    cdef cppclass asn1SccDepthMap:
        asn1SccTime ref_time
        asn1SccDepthMap_timestamps timestamps
        asn1SccPROJECTION_TYPE vertical_projection
        asn1SccPROJECTION_TYPE horizontal_projection
        asn1SccDepthMap_vertical_interval vertical_interval
        asn1SccDepthMap_horizontal_interval horizontal_interval
        uint32_t vertical_size
        uint32_t horizontal_size
        asn1SccDepthMap_distances distances
        asn1SccDepthMap_remissions remissions


cdef extern from "Types/C/Image.h":
    cdef cppclass asn1SccImage_mode_t:
        pass


cdef extern from "Types/C/Image.h" namespace "asn1SccImage_mode_t":
    cdef asn1SccImage_mode_t asn1Sccmode_undefined
    cdef asn1SccImage_mode_t asn1Sccmode_grayscale
    cdef asn1SccImage_mode_t asn1Sccmode_rgb
    cdef asn1SccImage_mode_t asn1Sccmode_uyvy
    cdef asn1SccImage_mode_t asn1Sccmode_bgr
    cdef asn1SccImage_mode_t asn1Sccmode_rgb32
    cdef asn1SccImage_mode_t asn1Sccraw_modes
    cdef asn1SccImage_mode_t asn1Sccmode_bayer
    cdef asn1SccImage_mode_t asn1Sccmode_bayer_rggb
    cdef asn1SccImage_mode_t asn1Sccmode_bayer_grbg
    cdef asn1SccImage_mode_t asn1Sccmode_bayer_bggr
    cdef asn1SccImage_mode_t asn1Sccmode_bayer_gbrg
    cdef asn1SccImage_mode_t asn1Scccompressed_modes
    cdef asn1SccImage_mode_t asn1SccImage_mode_t_mode_pjpg
    cdef asn1SccImage_mode_t asn1Sccmode_jpeg
    cdef asn1SccImage_mode_t asn1Sccmode_png


cdef extern from "Types/C/Image.h":
    cdef cppclass asn1SccImage_status_t:
        pass


cdef extern from "Types/C/Image.h" namespace "asn1SccImage_status_t":
    cdef asn1SccImage_status_t asn1Sccstatus_empty
    cdef asn1SccImage_status_t asn1Sccstatus_valid
    cdef asn1SccImage_status_t asn1Sccstatus_invalid


cdef extern from "Types/C/Image.h":
    cdef cppclass asn1SccImage_size_t:
        uint16_t width
        uint16_t height
    cdef cppclass asn1SccImage_attrib_t:
        asn1SccT_String data
        asn1SccT_String att_name
    cdef cppclass asn1SccImage_attributes:
        int nCount
        asn1SccImage_attrib_t[5] arr
    cdef cppclass asn1SccImage_image:
        int nCount
        unsigned char[24883200] arr
    cdef cppclass asn1SccImage:
        asn1SccTime frame_time
        asn1SccTime received_time
        asn1SccImage_image image
        asn1SccImage_attributes attributes
        asn1SccImage_size_t datasize
        uint32_t data_depth
        uint32_t pixel_size
        uint32_t row_size
        asn1SccImage_mode_t frame_mode
        asn1SccImage_status_t frame_status
    cdef cppclass asn1SccImagePair:
        asn1SccTime frame_time
        asn1SccImage first
        asn1SccImage second
        uint32_t id


cdef extern from "Types/C/Array3D.h":
    cdef cppclass asn1SccArray3D_depth_t:
        pass


cdef extern from "Types/C/Array3D.h" namespace "asn1SccArray3D_depth_t":
    cdef asn1SccArray3D_depth_t asn1Sccdepth_8U
    cdef asn1SccArray3D_depth_t asn1Sccdepth_8S
    cdef asn1SccArray3D_depth_t asn1Sccdepth_16U
    cdef asn1SccArray3D_depth_t asn1Sccdepth_16S
    cdef asn1SccArray3D_depth_t asn1Sccdepth_32S
    cdef asn1SccArray3D_depth_t asn1Sccdepth_32F
    cdef asn1SccArray3D_depth_t asn1Sccdepth_64F


cdef extern from "Types/C/Array3D.h":
    cdef cppclass asn1SccArray3D:
        uint32_t msgVersion
        uint32_t rows
        uint32_t cols
        uint32_t channels
        asn1SccArray3D_depth_t depth
        uint32_t rowSize
        asn1SccArray3D_data data
    cdef uint32_t array3D_Version
    cdef cppclass asn1SccArray3D_data:
        int nCount
        unsigned char[66355200] arr


cdef extern from "Types/C/Frame.h":
    cdef cppclass asn1SccFrame_errorType_t:
        pass


cdef extern from "Types/C/Frame.h" namespace "asn1SccFrame_errorType_t":
    cdef asn1SccFrame_errorType_t asn1Sccerror_UNDEFINED
    cdef asn1SccFrame_errorType_t asn1Sccerror_DEAD
    cdef asn1SccFrame_errorType_t asn1Sccerror_FILTERED


cdef extern from "Types/C/Frame.h":
    cdef cppclass asn1SccFrame_pixelModel_t:
        pass


cdef extern from "Types/C/Frame.h" namespace "asn1SccFrame_pixelModel_t":
    cdef asn1SccFrame_pixelModel_t asn1Sccpix_UNDEF
    cdef asn1SccFrame_pixelModel_t asn1Sccpix_POLY
    cdef asn1SccFrame_pixelModel_t asn1Sccpix_DISP


cdef extern from "Types/C/Frame.h":
    cdef cppclass asn1SccFrame_mode_t:
        pass


cdef extern from "Types/C/Frame.h" namespace "asn1SccFrame_mode_t":
    cdef asn1SccFrame_mode_t asn1Sccmode_UNDEF
    cdef asn1SccFrame_mode_t asn1Sccmode_GRAY
    cdef asn1SccFrame_mode_t asn1Sccmode_RGB
    cdef asn1SccFrame_mode_t asn1Sccmode_RGBA
    cdef asn1SccFrame_mode_t asn1Sccmode_BGR
    cdef asn1SccFrame_mode_t asn1Sccmode_BGRA
    cdef asn1SccFrame_mode_t asn1Sccmode_HSV
    cdef asn1SccFrame_mode_t asn1Sccmode_HLS
    cdef asn1SccFrame_mode_t asn1Sccmode_YUV
    cdef asn1SccFrame_mode_t asn1Sccmode_UYVY
    cdef asn1SccFrame_mode_t asn1Sccmode_Lab
    cdef asn1SccFrame_mode_t asn1Sccmode_Luv
    cdef asn1SccFrame_mode_t asn1Sccmode_XYZ
    cdef asn1SccFrame_mode_t asn1Sccmode_YCrCb
    cdef asn1SccFrame_mode_t asn1Sccmode_RGB32
    cdef asn1SccFrame_mode_t asn1Sccmode_Bayer_RGGB
    cdef asn1SccFrame_mode_t asn1Sccmode_Bayer_GRBG
    cdef asn1SccFrame_mode_t asn1Sccmode_Bayer_BGGR
    cdef asn1SccFrame_mode_t asn1Sccmode_Bayer_GBRG
    cdef asn1SccFrame_mode_t asn1Sccmode_PJPG
    cdef asn1SccFrame_mode_t asn1Sccmode_JPEG
    cdef asn1SccFrame_mode_t asn1Sccmode_PNG


cdef extern from "Types/C/Frame.h":
    cdef cppclass asn1SccFrame_status_t:
        pass


cdef extern from "Types/C/Frame.h" namespace "asn1SccFrame_status_t":
    cdef asn1SccFrame_status_t asn1Sccstatus_EMPTY
    cdef asn1SccFrame_status_t asn1Sccstatus_VALID
    cdef asn1SccFrame_status_t asn1Sccstatus_INVALID


cdef extern from "Types/C/Frame.h":
    cdef cppclass asn1SccFrame_cameraModel_t:
        pass


cdef extern from "Types/C/Frame.h" namespace "asn1SccFrame_cameraModel_t":
    cdef asn1SccFrame_cameraModel_t asn1Scccam_UNDEF
    cdef asn1SccFrame_cameraModel_t asn1Scccam_PINHOLE
    cdef asn1SccFrame_cameraModel_t asn1Scccam_FISHEYE
    cdef asn1SccFrame_cameraModel_t asn1Scccam_MAPS


cdef extern from "Types/C/Frame.h":
    cdef cppclass asn1SccFrame:
        uint32_t msgVersion
        asn1SccFrame_metadata_t metadata
        asn1SccFrame_intrinsic_t intrinsic
        asn1SccFrame_extrinsic_t extrinsic
        asn1SccArray3D data
    cdef uint32_t frame_Version
    cdef void asn1SccFrame_Initialize(asn1SccFrame*)
    cdef cppclass asn1SccFrame_metadata_t:
        uint32_t msgVersion
        asn1SccTime timeStamp
        asn1SccTime receivedTime
        asn1SccFrame_pixelModel_t pixelModel
        asn1SccVectorXd pixelCoeffs
        asn1SccFrame_metadata_t_errValues errValues
        asn1SccFrame_metadata_t_attributes attributes
        asn1SccFrame_mode_t mode
        asn1SccFrame_status_t status
    cdef cppclass asn1SccFrame_intrinsic_t:
        uint32_t msgVersion;
        asn1SccT_String sensorId;
        asn1SccMatrix3d cameraMatrix;
        asn1SccFrame_cameraModel_t cameraModel;
        asn1SccVectorXd distCoeffs;
    cdef cppclass asn1SccFrame_extrinsic_t:
        uint32_t msgVersion
        bool hasFixedTransform
        asn1SccTransformWithCovariance pose_robotFrame_sensorFrame
        asn1SccTransformWithCovariance pose_fixedFrame_robotFrame
    cdef cppclass asn1SccFrame_metadata_t_errValues:
        int nCount
        asn1SccFrame_error_t arr[3]
    cdef cppclass asn1SccFrame_metadata_t_attributes:
        int nCount
        asn1SccFrame_attrib_t arr[5]
    cdef cppclass asn1SccFrame_error_t:
        asn1SccFrame_errorType_t type
        double value
    cdef cppclass asn1SccFrame_attrib_t:
        asn1SccT_String name
        asn1SccT_String data

    cdef cppclass asn1SccFramePair:
        uint32_t msgVersion
        double baseline
        asn1SccFrame left
        asn1SccFrame right
    cdef void asn1SccFramePair_Initialize(asn1SccFramePair*)


cdef extern from "Types/C/Map.h":
    cdef cppclass asn1SccMap_type_t:
        pass


cdef extern from "Types/C/Map.h" namespace "asn1SccMap_type_t":
    cdef asn1SccMap_type_t asn1Sccmap_UNDEF
    cdef asn1SccMap_type_t asn1Sccmap_DEM
    cdef asn1SccMap_type_t asn1Sccmap_NAV


cdef extern from "Types/C/Map.h":
    cdef cppclass asn1SccMap:
        uint32_t msgVersion
        asn1SccMap_metadata_t metadata
        asn1SccArray3D data
    cdef uint32_t map_Version
    cdef void asn1SccMap_Initialize(asn1SccMap*)
    cdef cppclass asn1SccMap_metadata_t:
        uint32_t msgVersion
        asn1SccTime timeStamp
        asn1SccMap_type_t type
        asn1SccMap_metadata_t_errValues errValues
        double scale
        asn1SccTransformWithCovariance pose_fixedFrame_mapFrame
    cdef cppclass asn1SccMap_metadata_t_errValues:
        int nCount
        asn1SccFrame_error_t arr[5]
    bool asn1SccMap_Decode(asn1SccMap* pVal, BitStream* pBitStrm, int* pErrCode)
    int asn1SccMap_REQUIRED_BYTES_FOR_ENCODING


cdef extern from "Types/C/asn1crt.h":
    cdef cppclass BitStream:
        unsigned char* buf
        long count
        long currentByte
        int currentBit
    void BitStream_Init(BitStream* pBitStrm, unsigned char* buf, long count)
