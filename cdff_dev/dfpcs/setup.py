import os
import glob
import warnings
import numpy
from subprocess import Popen, PIPE
from cdff_dev.path import load_cdffpath


cdffpath = load_cdffpath()
DEFAULT_INCLUDE_DIRS = [
    ".",
    numpy.get_include(),
    # TODO move to installation folder:
    os.path.join(cdffpath, "Common"),
    os.path.join(cdffpath, "Common", "Converters"),
    os.path.join(cdffpath, "Common", "Types", "C"),
    os.path.join(cdffpath, "Common", "Types", "CPP"),
    os.path.join(cdffpath, "DFNs"),
    os.path.join(cdffpath, "DFPCs"),
    os.path.join(cdffpath, "Tools"),
    os.path.join(cdffpath, "CC"),
]
DEFAULT_LIBRARY_DIRS = [
    # TODO move to installation folder:
    os.path.join(cdffpath, "build"),
    os.path.join(cdffpath, "build", "Common", "Types"),
    os.path.join(cdffpath, "build", "Common", "Helpers"),
    os.path.join(cdffpath, "build", "Common", "Visualizers"),
    os.path.join(cdffpath, "build", "DFNs"),
    os.path.join(cdffpath, "build", "DFPCs"),
    os.path.join(cdffpath, "build", "CC"),
] + list(glob.glob(os.path.join(cdffpath, "build", "DFNs", "*")))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("dfpcs", parent_package, top_path)

    extra_compile_args = [
        "-std=c++11",
        "-O3",
        # disable warnings caused by Cython using the deprecated
        # NumPy C-API
        "-Wno-cpp", "-Wno-unused-function"
    ]


    autoproj_available = check_autoproj()
    if autoproj_available:
        make_reconstruction3d(config, cdffpath, extra_compile_args)

    return config


def check_autoproj():  # TODO refactor with main setup.py
    autoproj_available = "AUTOPROJ_CURRENT_ROOT" in os.environ
    if not autoproj_available:
        warnings.warn(
            "autoproj environment not detected, EnviRe will not be available",
            UserWarning)
    return autoproj_available


def get_include_dirs(libraries):
    output = pkgconfig("--cflags-only-I", libraries)
    # Remove -I
    return list(map(lambda dir: dir[2:], output.split()))


def get_library_dirs(libraries):
    output = pkgconfig("--libs-only-L", libraries)
    # Remove -L
    return list(map(lambda dir: dir[2:], output.split()))


def get_libraries(libraries):
    output = pkgconfig("--libs", libraries)
    # Remove -l
    return list(map(lambda dir: dir[2:], output.split()))


def pkgconfig(arg, libraries):
    p = Popen(["pkg-config", arg] + libraries,
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        raise IOError(err.decode("utf-8"))
    return output.decode("utf-8")


def find_ceres():
    return find_library("ceres")


def find_boost_system():
    return find_library("boost_system", "boost/system")


def find_library(name, expected_include_path=None):
    if expected_include_path is None:
        expected_include_path = name
    lib_install_dir = None
    if check_autoproj():
        autoproj_current_root = os.environ.get("AUTOPROJ_CURRENT_ROOT", None)
        install_dir = os.path.join(autoproj_current_root, "install")
        if os.path.exists(os.path.join(
                install_dir, "include", expected_include_path)):
            lib_install_dir = os.path.join(install_dir)
    if lib_install_dir is None:
        if os.path.exists("/usr/local/include/" + expected_include_path):
            lib_install_dir = "/usr/local"
        elif os.path.exists("/usr/include/" + expected_include_path):
            lib_install_dir = "/usr"
        else:
            raise RuntimeError("Could not find Ceres")
    return {
        "include_dirs": [os.path.join(lib_install_dir, "include",
                                      expected_include_path)],
        "library_dirs": [os.path.join(lib_install_dir, "lib")],
        "libraries": [name]
    }


def find_dependencies_of(libraries, cdffpath, recursive=True, blacklist=(),
                         cmake_cache=None, verbose=0):
    """Find dependencies of libraries.

    Parameters
    ----------
    libraries : list
        Names of libraries

    cdffpath : str
        Path to CDFF

    recursive : bool, optional (default: True)
        Search recursively

    blacklist : list, optional (default: ())
        Libraries that will be ignored

    Returns
    -------
    dependencies : list
        All dependencies. If parsed recursively, dependencies of dependencies
        come after dependencies. All libraries occur only once.
    """
    if cmake_cache is None:
        with open(os.path.join(cdffpath, "build", "CMakeCache.txt")) as f:
            cmake_cache = f.read()

    dependencies = []
    for library in libraries:
        if verbose >= 2:
            print("library: %s" % library)

        new_deps = _search_cmake_cache(library, cmake_cache)

        if verbose >= 2:
            print("direct dependencies: %s" % new_deps)

        filtered_deps = _apply_blacklist(new_deps, blacklist)

        if verbose >= 2:
            print("filtered direct dependencies: %s" % filtered_deps)

        if recursive:
            filtered_deps += find_dependencies_of(
                filtered_deps, cdffpath, recursive, blacklist, cmake_cache)
        dependencies += filtered_deps

    return _only_last_occurence(dependencies)


def _apply_blacklist(new_deps, blacklist):
    filtered_deps = []
    for dep in new_deps:
        ok = True
        for blacklisted in blacklist:
            if dep.startswith(blacklisted):
                ok = False
                break
        if ok:
            filtered_deps.append(dep)
    return filtered_deps


def _search_cmake_cache(library, cmake_cache=None):
    # search for a line similar to this one:
    # shot_descriptor_3d_LIB_DEPENDS:STATIC=general;feature_description_3d;...
    start = os.linesep + library + "_LIB_DEPENDS:STATIC="
    start_idx = cmake_cache.find(start) + len(start)
    end_idx = cmake_cache.find(os.linesep, start_idx)
    dependencies = cmake_cache[start_idx:end_idx].replace(
        "general;", "").split(";")[:-1]
    return dependencies


def _only_last_occurence(dependencies):
    unique = []
    for dep in dependencies:
        if dep in unique:
            unique.remove(dep)
        unique.append(dep)
    return unique


def make_reconstruction3d(config, cdffpath, extra_compile_args):
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
    dep_inc_dirs = get_include_dirs(libraries)
    dep_lib_dirs = get_library_dirs(libraries)
    dep_libs = get_libraries(libraries)

    ceres_info = find_ceres()
    dep_inc_dirs += ceres_info["include_dirs"]
    dep_lib_dirs += ceres_info["library_dirs"]
    dep_libs += ceres_info["libraries"]

    boost_system_info = find_boost_system()
    dep_inc_dirs += boost_system_info["include_dirs"]
    dep_lib_dirs += boost_system_info["library_dirs"]
    dep_libs += boost_system_info["libraries"]

    dfpc_libraries = [
        "adjustment_from_stereo",
        "dense_registration_from_stereo",
        "sparse_registration_from_stereo",
        "reconstruction_from_motion",
        "reconstruction_from_stereo",
        "registration_from_stereo",
        "estimation_from_stereo",
    ]
    dfpc_deps = find_dependencies_of(
        dfpc_libraries, cdffpath, blacklist=("pcl", "vtk", "verdict"))
    print(dfpc_deps)

    config.add_extension(
        "reconstruction3d",
        sources=["reconstruction3d.pyx"],
        include_dirs=[
            os.path.join(cdffpath, "DFPCs", "Reconstruction3D"),
        ] + DEFAULT_INCLUDE_DIRS + dep_inc_dirs,
        library_dirs=[
            # TODO move to installation folder:
            os.path.join(cdffpath, "build", "DFPCs", "Reconstruction3D"),
        ] + DEFAULT_LIBRARY_DIRS + dep_lib_dirs,
        # make sure that libraries used in other libraries are linked last!
        # for example: an implementation must be linked before its interface
        libraries=dfpc_libraries + dfpc_deps + dep_libs,
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
