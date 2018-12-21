#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from distutils.sysconfig import get_config_vars
from distutils.command.clean import clean
from Cython.Build import cythonize
import numpy
import glob
import cdff_dev
import build_tools
from cdff_dev.path import load_cdffpath, CTYPESDIR


def strict_prototypes_workaround():
    # Workaround to remove '-Wstrict-prototypes' from compiler invocation
    opt = get_config_vars('OPT')[0]
    os.environ['OPT'] = " ".join(flag for flag in opt.split()
                                 if flag != '-Wstrict-prototypes')


# Custom clean command to remove build artifacts

class CleanCommand(clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        clean.run(self)

        print("removing Cython build artifacts")
        cwd = os.path.abspath(os.path.dirname(__file__))
        dirs = [cwd, os.path.join(cwd, "cdff_dev", "extensions", "*"),
                os.path.join(cwd, "cdff_dev", "dfpcs"),
                #os.path.join(cwd, "cdff_dev", "dfns")
                ]
        for path in dirs:
            filenames = (glob.glob(os.path.join(path, "*.cpp")) +
                         glob.glob(os.path.join(path, "*.so")) +
                         glob.glob(os.path.join(path, "*.pyd")))
            for filename in filenames:
                print(filename)
                os.remove(filename)


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage("cdff_dev")

    autoproj_available = build_tools.check_autoproj()

    cdff_types_files = ["_cdff_types.pxd", "cdff_types.pxd", "cdff_types.pyx"]
    cdff_envire_files = ["_cdff_envire.pxd", "cdff_envire.pxd",
                         "cdff_envire.pyx"]
    cython_files = cdff_types_files
    if autoproj_available:
        cython_files += cdff_envire_files
    cython_files = [(".", filename) for filename in cython_files]

    config.add_data_files(*cython_files)

    # uncomment to see more outputs from the linker
    #os.environ["CFLAGS"] = "-Xlinker -v"

    cdffpath = load_cdffpath()
    typespath = os.path.join(cdffpath, "Common")

    make_cdff_types(config, cdffpath, typespath)
    if autoproj_available:
        make_cdff_envire(config, typespath)

    config.ext_modules = cythonize(config.ext_modules)

    return config


def make_cdff_types(config, cdffpath, typespath):
    config.add_extension(
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


def make_cdff_envire(config, typespath):
    autoproj_current_root = os.environ.get("AUTOPROJ_CURRENT_ROOT", None)
    install_dir = os.path.join(autoproj_current_root, "install")
    # this path is currently only used in CI image:
    eigen_include_dir = "/usr/local/include/eigen3"
    if not os.path.exists(eigen_include_dir):
        eigen_include_dir = os.path.join(install_dir, "include", "eigen3")
        print("using Eigen 3 from autoproj installation")
    if not os.path.exists(eigen_include_dir):
        eigen_include_dir = "/usr/include/eigen3/"
        print("using Eigen 3 from system path")
    config.add_extension(
        "cdff_envire",
        sources=["cdff_envire.pyx"],
        include_dirs=[
            ".",
            "cpp_helpers",
            numpy.get_include(),
            os.path.join(install_dir, "include"),
            eigen_include_dir,
            typespath
        ] + build_tools.DEFAULT_INCLUDE_DIRS,
        library_dirs=[
            os.path.join(install_dir, "lib")
        ] + build_tools.DEFAULT_LIBRARY_DIRS,
        libraries=["cdff_types", "base-types", "envire_core", "envire_urdf",
                   "urdfdom_model", "envire_visualizer_interface"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=build_tools.extra_compile_args
    )


if __name__ == "__main__":
    from numpy.distutils.core import setup

    strict_prototypes_workaround()
    metadata = dict(
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
        packages=['cdff_dev'],
        package_data={'cdff_dev': ['templates/*.template']},
        requires=['pyyaml', 'cython', 'Jinja2', 'numpy', 'pydot'],
        cmdclass = {'clean': CleanCommand},
        configuration=configuration
    )
    setup(**metadata)
