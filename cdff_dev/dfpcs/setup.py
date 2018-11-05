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
        make_reconstruction3d(
            config, cdffpath, build_tools.extra_compile_args)
        make_pointcloudmodellocalisation(
            config, cdffpath, build_tools.extra_compile_args)

    return config


def make_reconstruction3d(config, cdffpath, extra_compile_args):
    libraries = ["opencv", "eigen3", "yaml-cpp", "pcl_common-1.8",
                 "pcl_visualization-1.8"]

    # use pkg-config for external dependencies
    dep_inc_dirs = build_tools.get_include_dirs(libraries)

    dfpc_libraries = [
        "adjustment_from_stereo",
        "dense_registration_from_stereo",
        "sparse_registration_from_stereo",
        "reconstruction_from_motion",
        "reconstruction_from_stereo",
        "registration_from_stereo",
        "estimation_from_stereo",
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
        extra_compile_args=extra_compile_args
    )


def make_pointcloudmodellocalisation(config, cdffpath, extra_compile_args):
    config.add_extension(
        "pointcloudmodellocalisation",
        sources=["pointcloudmodellocalisation.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFPCs", "PointCloudModelLocalisation"),
        ] + build_tools.DEFAULT_INCLUDE_DIRS,
        library_dirs=[
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFPCs", "PointCloudModelLocalisation"),
        ],
        # make sure that libraries used in other libraries are linked last!
        # for example: an implementation must be linked before its interface
        libraries=["dfpc_implementation_features_matching_3d"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
