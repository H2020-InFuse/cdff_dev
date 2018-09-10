import os
import warnings
import numpy
from cdff_dev.path import load_cdffpath, CTYPESDIR


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("reconstruction3d", parent_package, top_path)

    extra_compile_args = [
        "-std=c++11",
        "-O3",
        # disable warnings caused by Cython using the deprecated
        # NumPy C-API
        "-Wno-cpp", "-Wno-unused-function"
    ]

    cdffpath = load_cdffpath()

    autoproj_available = check_autoproj()
    if autoproj_available:
        make_dfpc(config, cdffpath, extra_compile_args)

    return config


def check_autoproj():  # TODO refactor with main setup.py
    autoproj_available = "AUTOPROJ_CURRENT_ROOT" in os.environ
    if not autoproj_available:
        warnings.warn(
            "autoproj environment not detected, EnviRe will not be available",
            UserWarning)
    return autoproj_available


def make_dfpc(config, cdffpath, extra_compile_args):
    autoproj_current_root = os.environ.get("AUTOPROJ_CURRENT_ROOT", None)
    install_dir = os.path.join(autoproj_current_root, "install")
    config.add_extension(
        "reconstruction3d",
        sources=["reconstruction3d.pyx"],
        include_dirs=[
            ".",
            numpy.get_include(),
            # TODO find automatically
            os.path.join(install_dir, "include", "pcl-1.8"),
            os.path.join(install_dir, "include", "eigen3"),
            # TODO move to installation folder:
            os.path.join(cdffpath, "Common"),
            os.path.join(cdffpath, "Common", "Converters"),
            os.path.join(cdffpath, "Common", "Types", "C"),
            os.path.join(cdffpath, "Common", "Types", "CPP"),
            os.path.join(cdffpath, "DFNs"),
            os.path.join(cdffpath, "DFPCs"),
            os.path.join(cdffpath, "DFPCs", "Reconstruction3D"),
            os.path.join(cdffpath, "Tools"),
            os.path.join(cdffpath, "CC"),
        ],
        library_dirs=[
            os.path.join(install_dir, "lib"),
            # TODO find yaml-cpp
            # TODO move to installation folder:
            os.path.join(cdffpath, "build"),
            # for cdff_types
            os.path.join(cdffpath, "build", "Common", "Types"),
            # for cdff_helpers
            os.path.join(cdffpath, "build", "Common", "Helpers"),
            # for cdff_opencv_visualizer, cdff_pcl_visualizer
            os.path.join(cdffpath, "build", "Common", "Visualizers"),
            # for dfpc_configurator
            os.path.join(cdffpath, "build", "DFPCs"),
        ],
        libraries=["cdff_types", "cdff_helpers", "cdff_opencv_visualizer",
                   "cdff_pcl_visualizer"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
