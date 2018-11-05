import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import build_tools


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("pcl", parent_package, top_path)

    libraries = ["opencv", "eigen3", "pcl_common-1.8", "pcl_io-1.8"]

    # use pkg-config for external dependencies
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)
    dep_libs = build_tools.get_libraries(libraries)

    helper_libraries = [
        "cdff_converters",
        "cdff_logger",
        "cdff_types"
    ]

    config.add_extension(
        "helpers",
        sources=["helpers.pyx"],
        include_dirs=["cpp_helpers"] + build_tools.DEFAULT_INCLUDE_DIRS +
                     dep_inc_dirs,
        library_dirs=build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=helper_libraries + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
