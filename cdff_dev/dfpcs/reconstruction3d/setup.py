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
            os.path.join(cdffpath, "DFPCs"),
            os.path.join(cdffpath, "DFPCs", "Reconstruction3D"),
            os.path.join(cdffpath, CTYPESDIR),
        ],
        library_dirs=[
            os.path.join(install_dir, "lib")
        ],
        libraries=["base-types", "envire_core", "envire_urdf",
                   "urdfdom_model", "envire_visualizer_interface"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
