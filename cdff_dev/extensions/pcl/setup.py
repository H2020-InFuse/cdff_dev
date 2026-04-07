import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import build_tools
from setuptools import Extension


def get_extensions():
    # PCL 1.14 pkg-config names no longer carry the version suffix
    libraries = ["opencv", "eigen3", "pcl_common", "pcl_io"]

    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)
    dep_libs = build_tools.get_libraries(libraries)

    helper_libraries = [
        "cdff_converters",
        "cdff_logger",
        "cdff_types"
    ]

    return [Extension(
        "cdff_dev.extensions.pcl.helpers",
        sources=["cdff_dev/extensions/pcl/helpers.pyx"],
        include_dirs=["cpp_helpers"] + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=helper_libraries + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )]