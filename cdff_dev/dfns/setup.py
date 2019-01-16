import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import build_tools
from cdff_dev.path import load_cdffpath


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("dfns", parent_package, top_path)

    cdffpath = load_cdffpath()
    autoproj_available = build_tools.check_autoproj()
    if autoproj_available:
        make_imagedegradation(config, cdffpath)
        make_imagepairdegradation(config, cdffpath)
        make_disparityimage(config, cdffpath)
        make_disparitytopointcloud(config, cdffpath)
        # only edres version available at the moment
        #make_disparityfiltering(config, cdffpath)

    return config


def make_imagedegradation(config, cdffpath):
    libraries = ["opencv"]

    # use pkg-config for external dependencies
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)

    dep_libs = ["opencv_core", "opencv_imgproc"]
    # Edres is currently not publicly available.
    #edres_info = build_tools.find_library("edres-wrapper")
    #dep_inc_dirs += edres_info["include_dirs"]
    #dep_lib_dirs += edres_info["library_dirs"]
    #dep_libs += ["edres-wrapper"]

    dfn_libraries = [
        "cdff_dfn_image_degradation",
    ]

    config.add_extension(
        "imagedegradation",
        sources=["imagedegradation.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFNs", "ImageDegradation"),
        ] + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFNs", "ImageDegradation"),
        ] + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=dfn_libraries + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )


def make_imagepairdegradation(config, cdffpath):
    libraries = ["opencv"]

    # use pkg-config for external dependencies
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)

    dep_libs = []
    # Edres is currently not publicly available.
    #edres_info = build_tools.find_library("edres-wrapper")
    #dep_inc_dirs += edres_info["include_dirs"]
    #dep_lib_dirs += edres_info["library_dirs"]
    #dep_libs += ["edres-wrapper"]

    dfn_libraries = [
        "cdff_dfn_stereo_degradation",
    ]

    config.add_extension(
        "stereodegradation",
        sources=["stereodegradation.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFNs", "StereoDegradation"),
        ] + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFNs", "StereoDegradation"),
        ] + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=dfn_libraries + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )


def make_disparityimage(config, cdffpath):
    libraries = ["opencv"]

    # use pkg-config for external dependencies
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)

    dep_libs = []
    # Edres is currently not publicly available.
    #edres_info = build_tools.find_library("edres-wrapper")
    #dep_inc_dirs += edres_info["include_dirs"]
    #dep_lib_dirs += edres_info["library_dirs"]
    #dep_libs += ["edres-wrapper"]

    dfn_libraries = [
        "cdff_dfn_disparity_image",
    ]

    config.add_extension(
        "disparityimage",
        sources=["disparityimage.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFNs", "DisparityImage"),
        ] + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFNs", "DisparityImage"),
        ] + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=dfn_libraries + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )


def make_disparitytopointcloud(config, cdffpath):
    libraries = ["opencv"]

    # use pkg-config for external dependencies
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)

    dep_libs = []
    # Edres is currently not publicly available.
    #edres_info = build_tools.find_library("edres-wrapper")
    #dep_inc_dirs += edres_info["include_dirs"]
    #dep_lib_dirs += edres_info["library_dirs"]
    #dep_libs += ["edres-wrapper"]

    dfn_libraries = [
        "cdff_dfn_disparity_to_pointcloud",
    ]

    config.add_extension(
        "disparitytopointcloud",
        sources=["disparitytopointcloud.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFNs", "DisparityToPointCloud"),
        ] + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFNs", "DisparityToPointCloud"),
        ] + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=dfn_libraries + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )


def make_disparityfiltering(config, cdffpath):
    libraries = ["opencv"]

    # use pkg-config for external dependencies
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)

    dep_libs = []
    # Edres is currently not publicly available.
    edres_info = build_tools.find_library("edres-wrapper")
    dep_inc_dirs += edres_info["include_dirs"]
    dep_lib_dirs += edres_info["library_dirs"]
    dep_libs += ["edres-wrapper"]

    dfn_libraries = [
        "cdff_dfn_disparity_filtering",
    ]

    import glob
    print(list(glob.glob(os.path.join(cdffpath, "DFNs", "DisparityFiltering") + "/*.hpp")))

    config.add_extension(
        "disparityfiltering",
        sources=["disparityfiltering.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFNs", "DisparityFiltering"),
        ] + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFNs", "DisparityFiltering"),
        ] + build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=dfn_libraries + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
