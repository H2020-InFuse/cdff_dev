import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import build_tools
from cdff_dev.path import load_cdffpath
from setuptools import Extension


def get_extensions():
    if not build_tools.check_autoproj():
        return []

    cdffpath = load_cdffpath()
    extensions = []
    extensions += _make_imagedegradation(cdffpath)
    extensions += _make_imagepairdegradation(cdffpath)
    extensions += _make_disparityimage(cdffpath)
    extensions += _make_disparitytopointcloud(cdffpath)
    extensions += _make_disparitytopointcloudwithintensity(cdffpath)
    return extensions


def _make_imagedegradation(cdffpath):
    libraries = ["opencv"]
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)
    dep_libs = ["opencv_core", "opencv_imgproc"]

    return [Extension(
        "cdff_dev.dfns.imagedegradation",
        sources=["cdff_dev/dfns/imagedegradation.pyx"],
        include_dirs=[os.path.join(cdffpath, "DFNs", "ImageDegradation")]
            + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[os.path.join(cdffpath, "build", "DFNs", "ImageDegradation")]
            + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=["cdff_dfn_image_degradation"] + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )]


def _make_imagepairdegradation(cdffpath):
    libraries = ["opencv"]
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)

    return [Extension(
        "cdff_dev.dfns.stereodegradation",
        sources=["cdff_dev/dfns/stereodegradation.pyx"],
        include_dirs=[os.path.join(cdffpath, "DFNs", "StereoDegradation")]
            + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[os.path.join(cdffpath, "build", "DFNs", "StereoDegradation")]
            + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=["cdff_dfn_stereo_degradation"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )]


def _make_disparityimage(cdffpath):
    libraries = ["opencv"]
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)

    return [Extension(
        "cdff_dev.dfns.disparityimage",
        sources=["cdff_dev/dfns/disparityimage.pyx"],
        include_dirs=[os.path.join(cdffpath, "DFNs", "DisparityImage")]
            + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[os.path.join(cdffpath, "build", "DFNs", "DisparityImage")]
            + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=["cdff_dfn_disparity_image"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )]


def _make_disparitytopointcloud(cdffpath):
    libraries = ["opencv"]
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)

    return [Extension(
        "cdff_dev.dfns.disparitytopointcloud",
        sources=["cdff_dev/dfns/disparitytopointcloud.pyx"],
        include_dirs=[os.path.join(cdffpath, "DFNs", "DisparityToPointCloud")]
            + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[os.path.join(cdffpath, "build", "DFNs", "DisparityToPointCloud")]
            + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=["cdff_dfn_disparity_to_pointcloud"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )]


def _make_disparitytopointcloudwithintensity(cdffpath):
    libraries = ["opencv"]
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)

    return [Extension(
        "cdff_dev.dfns.disparitytopointcloudwithintensity",
        sources=["cdff_dev/dfns/disparitytopointcloudwithintensity.pyx"],
        include_dirs=[os.path.join(cdffpath, "DFNs", "DisparityToPointCloudWithIntensity")]
            + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[os.path.join(cdffpath, "build", "DFNs", "DisparityToPointCloudWithIntensity")]
            + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=["cdff_dfn_disparity_to_pointcloud_with_intensity"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )]