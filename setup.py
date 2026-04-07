#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import glob
import cdff_dev
import build_tools
from cdff_dev.path import load_cdffpath, CTYPESDIR
from cdff_dev.setup import get_extensions as get_cdff_dev_extensions


def make_cdff_types(cdffpath, typespath):
    return Extension(
        "cdff_types",
        sources=["cdff_types.pyx"],
        include_dirs=[
            ".",
            "cpp_helpers",
            numpy.get_include(),
            typespath,
            os.path.join(cdffpath, "Common/Types/CPP")
        ],
        library_dirs=[
            os.path.join(cdffpath, "build", "Common", "Types"),
            os.path.join(cdffpath, "build", "Common", "Loggers")
        ],
        libraries=["cdff_logger", "cdff_types"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )


def make_cdff_envire(typespath):
    autoproj_current_root = os.environ.get("AUTOPROJ_CURRENT_ROOT", None)
    install_dir = os.path.join(autoproj_current_root, "install")
    eigen_include_dir = "/usr/local/include/eigen3"
    if not os.path.exists(eigen_include_dir):
        eigen_include_dir = os.path.join(install_dir, "include", "eigen3")
        print("using Eigen 3 from autoproj installation")
    if not os.path.exists(eigen_include_dir):
        eigen_include_dir = "/usr/include/eigen3/"
        print("using Eigen 3 from system path")
    return Extension(
        "cdff_envire",
        sources=["cdff_envire.pyx"],
        include_dirs=[
            ".",
            "cpp_helpers",
            numpy.get_include(),
            os.path.join(install_dir, "include"),
            os.path.join(install_dir, "include", "urdfdom"),
            os.path.join(install_dir, "include", "urdfdom_headers"),
            eigen_include_dir,
            typespath
        ] + build_tools.DEFAULT_INCLUDE_DIRS,
        library_dirs=[
            os.path.join(install_dir, "lib")
        ] + build_tools.DEFAULT_LIBRARY_DIRS,
        libraries=["cdff_types", "base-types", "envire_core", "envire_urdf",
                   "urdfdom_model", "envire_visualizer_interface-qt5"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )


if __name__ == "__main__":
    cdffpath = load_cdffpath()
    typespath = os.path.join(cdffpath, "Common")
    autoproj_available = build_tools.check_autoproj()

    extensions = [make_cdff_types(cdffpath, typespath)]
    if autoproj_available:
        extensions.append(make_cdff_envire(typespath))

    extensions += get_cdff_dev_extensions()

    setup(
        name="cdff_dev",
        version=cdff_dev.__version__,
        description=cdff_dev.__description__,
        long_description=open("README.md").read(),
        scripts=[os.path.join("bin", "dfn_template_generator"),
                 os.path.join("bin", "dfpc_template_generator"),
                 os.path.join("bin", "cdff_dev_loginfo"),
                 os.path.join("bin", "cdff_dev_logshow"),
                 os.path.join("bin", "cdff_dev_chunk"),
                 os.path.join("bin", "cdff_dev_build_log_index"),
                 os.path.join("bin", "pyspace_export"),
                 os.path.join("bin", "dfpc_diagram")],
        packages=find_packages(exclude=["test", "tests"]),
        package_data={'cdff_dev': ['templates/*.template']},
        install_requires=['pyyaml', 'Jinja2', 'numpy', 'pydot'],
        ext_modules=cythonize(extensions, language_level=3, include_path=[
        ".",
        "cdff_dev/dfns",
        "cdff_dev/dfpcs",
        "cdff_dev/extensions/gps",
        "cdff_dev/extensions/pcl",
    ]),
    )