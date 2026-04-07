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
    return _make_reconstruction3d(cdffpath)


def _make_reconstruction3d(cdffpath):
    # PCL 1.14 pkg-config names no longer carry the version suffix
    libraries = ["opencv", "eigen3", "yaml-cpp", "pcl_common", "pcl_visualization"]

    dep_inc_dirs = build_tools.get_include_dirs(libraries)

    return [Extension(
        "cdff_dev.dfpcs.reconstruction3d",
        sources=["cdff_dev/dfpcs/reconstruction3d.pyx"],
        include_dirs=[os.path.join(cdffpath, "DFPCs", "Reconstruction3D")]
            + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[os.path.join(cdffpath, "build", "DFPCs", "Reconstruction3D")]
            + build_tools.DEFAULT_LIBRARY_DIRS,
        libraries=["cdff_dfpc_reconstruction_3d"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )]