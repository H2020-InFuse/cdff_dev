import os
import glob
import warnings
import numpy
from subprocess import Popen, PIPE
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


def get_include_dirs(libraries):
    output = pkgconfig("--cflags-only-I", libraries)
    # Remove -I
    return list(map(lambda dir: dir[2:], output.split()))


def get_library_dirs(libraries):
    output = pkgconfig("--libs-only-L", libraries)
    # Remove -L
    return list(map(lambda dir: dir[2:], output.split()))


def get_libraries(libraries):
    output = pkgconfig("--libs", libraries)
    # Remove -l
    return list(map(lambda dir: dir[2:], output.split()))


def pkgconfig(arg, libraries):
    p = Popen(["pkg-config", arg] + libraries,
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        raise IOError(err.decode("utf-8"))
    return output.decode("utf-8")


def find_ceres():
    return find_library("ceres")


def find_boost_system():
    return find_library("boost_system", "boost/system")


def find_library(name, expected_include_path=None):
    if expected_include_path is None:
        expected_include_path = name
    lib_install_dir = None
    if check_autoproj():
        autoproj_current_root = os.environ.get("AUTOPROJ_CURRENT_ROOT", None)
        install_dir = os.path.join(autoproj_current_root, "install")
        if os.path.exists(os.path.join(
                install_dir, "include", expected_include_path)):
            lib_install_dir = os.path.join(install_dir)
    if lib_install_dir is None:
        if os.path.exists("/usr/local/include/" + expected_include_path):
            lib_install_dir = "/usr/local"
        elif os.path.exists("/usr/include/" + expected_include_path):
            lib_install_dir = "/usr"
        else:
            raise RuntimeError("Could not find Ceres")
    return {
        "include_dirs": [os.path.join(lib_install_dir, "include",
                                      expected_include_path)],
        "library_dirs": [os.path.join(lib_install_dir, "lib")],
        "libraries": [name]
    }


def make_reconstruction3d(config, cdffpath, extra_compile_args):
    libraries = ["opencv", "eigen3", "yaml-cpp"]
    libraries += list(map(
        lambda lib: lib + "-1.8",
        ["pcl_common", "pcl_features", "pcl_search", "pcl_filters",
         "pcl_visualization", "pcl_io", "pcl_ml", "pcl_octree",
         "pcl_outofcore", "pcl_kdtree", "pcl_tracking", "pcl_stereo",
         "pcl_recognition", "pcl_registration", "pcl_people", "pcl_keypoints",
         "pcl_surface", "pcl_segmentation", "pcl_sample_consensus",
         "pcl_stereo"]))

    # use pkg-config for external dependencies
    dep_inc_dirs = get_include_dirs(libraries)
    dep_lib_dirs = get_library_dirs(libraries)
    dep_libs = get_libraries(libraries)

    ceres_info = find_ceres()
    dep_inc_dirs += ceres_info["include_dirs"]
    dep_lib_dirs += ceres_info["library_dirs"]
    dep_libs += ceres_info["libraries"]

    boost_system_info = find_boost_system()
    dep_inc_dirs += boost_system_info["include_dirs"]
    dep_lib_dirs += boost_system_info["library_dirs"]
    dep_libs += boost_system_info["libraries"]

    config.add_extension(
        "reconstruction3d",
        sources=["reconstruction3d.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFPCs", "Reconstruction3D"),
        ] + DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFPCs", "Reconstruction3D"),
        ] + DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
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
        ] + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
