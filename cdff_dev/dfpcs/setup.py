import os
import glob
import warnings
import numpy
from cdff_dev.path import load_cdffpath


cdffpath = load_cdffpath()
DEFAULT_INCLUDE_DIRS = [
    ".",
    numpy.get_include(),
    # TODO move to installation folder:
    os.path.join(cdffpath, "Common"),
    os.path.join(cdffpath, "Common", "Converters"),
    os.path.join(cdffpath, "Common", "Types", "C"),
    os.path.join(cdffpath, "Common", "Types", "CPP"),
    os.path.join(cdffpath, "DFNs"),
    os.path.join(cdffpath, "DFPCs"),
    os.path.join(cdffpath, "Tools"),
    os.path.join(cdffpath, "CC"),
]
DEFAULT_LIBRARY_DIRS = [
    # TODO move to installation folder:
    os.path.join(cdffpath, "build"),
    os.path.join(cdffpath, "build", "Common", "Types"),
    os.path.join(cdffpath, "build", "Common", "Helpers"),
    os.path.join(cdffpath, "build", "Common", "Visualizers"),
    os.path.join(cdffpath, "build", "DFNs"),
    os.path.join(cdffpath, "build", "DFPCs"),
    os.path.join(cdffpath, "build", "CC"),
] + list(glob.glob(os.path.join(cdffpath, "build", "DFNs", "*")))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("dfpcs", parent_package, top_path)

    extra_compile_args = [
        "-std=c++11",
        "-O3",
        # disable warnings caused by Cython using the deprecated
        # NumPy C-API
        "-Wno-cpp", "-Wno-unused-function"
    ]


    autoproj_available = check_autoproj()
    if autoproj_available:
        make_reconstruction3d(config, cdffpath, extra_compile_args)

    return config


def check_autoproj():  # TODO refactor with main setup.py
    autoproj_available = "AUTOPROJ_CURRENT_ROOT" in os.environ
    if not autoproj_available:
        warnings.warn(
            "autoproj environment not detected, EnviRe will not be available",
            UserWarning)
    return autoproj_available


def make_reconstruction3d(config, cdffpath, extra_compile_args):
    autoproj_current_root = os.environ.get("AUTOPROJ_CURRENT_ROOT", None)
    install_dir = os.path.join(autoproj_current_root, "install")

    config.add_extension(
        "reconstruction3d",
        sources=["reconstruction3d.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFPCs", "Reconstruction3D"),
            # TODO find automatically
            os.path.join(install_dir, "include", "pcl-1.8"),
            "/usr/local/include/pcl-1.8",
            os.path.join(install_dir, "include", "eigen3"),
            "/usr/local/include/eigen3",
        ] + DEFAULT_INCLUDE_DIRS,
        library_dirs=[
            # TODO find yaml-cpp, opencv, pcl, boost-system
            os.path.join(install_dir, "lib"),
            "/usr/local/lib",
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFPCs", "Reconstruction3D"),
        ] + DEFAULT_LIBRARY_DIRS,
        libraries=[
            # make sure that libraries used in other libraries are linked last!
            # for example: an implementation must be linked before its interface
            # TODO recursively parse CMakeCache.txt for <lib name>_LIB_DEPENDS

            "adjustment_from_stereo",
            "dense_registration_from_stereo",
            "sparse_registration_from_stereo",
            "reconstruction_from_motion",
            "reconstruction_from_stereo",
            "registration_from_stereo",
            "estimation_from_stereo",
            "reconstruction_3d",

            "dfpc_configurator",
            "dfns_builder",

            "fundamental_matrix_ransac",
            "fundamental_matrix_computation",
            "disparity_mapping",
            "hirschmuller_disparity_mapping",
            "scanline_optimization",
            "stereoReconstruction",
            "transform_3d_estimation_ceres",
            "transform_3d_estimation",
            "registration_icp_cc",
            "registration_icp_3d",
            "registration_3d",
            "svd_decomposition",
            "bundle_adjustment",
            "flann_matcher",
            "feature_matching_2d",
            "iterative_pnp_solver",
            "perspective_n_point_solving",
            "image_undistortion_rectification",
            "image_undistortion",
            "image_filtering",
            "convolution_filter",
            "depth_filtering",
            "ransac_3d",
            "icp_3d",
            "feature_matching_3d",
            "ceres_adjustment",
            "essential_matrix_decomposition",
            "cameras_transform_estimation",
            "orb_descriptor",
            "feature_description_2d",
            "triangulation",
            "point_cloud_reconstruction_2d_to_3d",
            "orb_detector_descriptor",
            "harris_detector_2d",
            "feature_extraction_2d",
            "transform_3d_estimation_least_squares_minimization",
            "harris_detector_3d",
            "feature_extraction_3d",
            "shot_descriptor_3d",
            "feature_description_3d",
            "cartesian_system_transform",
            "point_cloud_transform",
            "neighbour_point_average",
            "point_cloud_assembly",
            "hu_invariants",
            "primitive_matching",
            "octree",
            "voxelization",

            "cdff_types",
            "cdff_helpers",
            "cdff_opencv_visualizer",
            "cdff_pcl_visualizer",
            "cdff_logger",
            "converters_opencv",

            "cc_core_lib",

            "pcl_common",
            "pcl_features",
            "pcl_search",
            "pcl_filters",
            "pcl_visualization",
            "pcl_io",
            "pcl_ml",
            "pcl_io_ply",
            "pcl_octree",
            "pcl_outofcore",
            "pcl_kdtree",
            "pcl_tracking",
            "pcl_stereo",
            "pcl_recognition",
            "pcl_registration",
            "pcl_people",
            "pcl_keypoints",
            "pcl_surface",
            "pcl_segmentation",
            "pcl_sample_consensus",
            "pcl_stereo",

            "yaml-cpp",

            "boost_system",

            "ceres",

            "opencv_imgcodecs",
            "opencv_highgui",
            "opencv_imgcodecs",
            "opencv_features2d",
            "opencv_imgproc",
            "opencv_calib3d",
            "opencv_core",
        ],
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
