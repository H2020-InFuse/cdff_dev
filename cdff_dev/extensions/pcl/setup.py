import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import build_tools


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("pcl", parent_package, top_path)

    libraries = ["opencv", "eigen3", "yaml-cpp"]
    libraries += list(map(
        lambda lib: lib + "-1.8",
        ["pcl_common", "pcl_features", "pcl_search", "pcl_filters",
         "pcl_visualization", "pcl_io", "pcl_ml", "pcl_octree",
         "pcl_outofcore", "pcl_kdtree", "pcl_tracking", "pcl_stereo",
         "pcl_recognition", "pcl_registration", "pcl_people", "pcl_keypoints",
         "pcl_surface", "pcl_segmentation", "pcl_sample_consensus",
         "pcl_stereo"]))

    # use pkg-config for external dependencies
    dep_inc_dirs = build_tools.get_include_dirs(libraries)
    dep_lib_dirs = build_tools.get_library_dirs(libraries)
    dep_libs = build_tools.get_libraries(libraries)

    ceres_info = build_tools.find_ceres()
    dep_inc_dirs += ceres_info["include_dirs"]
    dep_lib_dirs += ceres_info["library_dirs"]
    dep_libs += ceres_info["libraries"]

    boost_system_info = build_tools.find_boost_system()
    dep_inc_dirs += boost_system_info["include_dirs"]
    dep_lib_dirs += boost_system_info["library_dirs"]
    dep_libs += boost_system_info["libraries"]

    cdffpath = build_tools.load_cdffpath()

    helper_libraries = [
        "converters_opencv",
        "cdff_types"
    ]
    helper_deps = build_tools.find_dependencies_of(
        helper_libraries, cdffpath, blacklist=("pcl", "vtk", "verdict"))

    config.add_extension(
        "helpers",
        sources=["helpers.pyx"],
        include_dirs=["cpp_helpers"] + build_tools.DEFAULT_INCLUDE_DIRS +
                     dep_inc_dirs,
        library_dirs=build_tools.DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        libraries=helper_libraries + helper_deps + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
