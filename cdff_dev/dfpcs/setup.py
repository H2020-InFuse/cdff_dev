import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import build_tools
from cdff_dev.path import load_cdffpath


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("dfpcs", parent_package, top_path)

    cdffpath = load_cdffpath()
    autoproj_available = build_tools.check_autoproj()
    if autoproj_available:
        make_reconstruction3d(config, cdffpath)

    return config


def make_reconstruction3d(config, cdffpath):
    libraries = ["opencv", "eigen3", "yaml-cpp", "pcl_common-1.8",
                 "pcl_visualization-1.8"]

    # use pkg-config for external dependencies
    dep_inc_dirs = build_tools.get_include_dirs(libraries)

    dfpc_libraries = [
        "cdff_dfpc_reconstruction_3d",
    ]

    config.add_extension(
        "reconstruction3d",
        sources=["reconstruction3d.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFPCs", "Reconstruction3D"),
        ] + build_tools.DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFPCs", "Reconstruction3D"),
        ] + build_tools.DEFAULT_LIBRARY_DIRS,
        # make sure that libraries used in other libraries are linked last!
        # for example: an implementation must be linked before its interface
        libraries=dfpc_libraries,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
