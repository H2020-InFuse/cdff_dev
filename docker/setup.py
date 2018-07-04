#!/usr/bin/env python3
from distutils.sysconfig import get_config_vars
from distutils.command.clean import clean
from Cython.Build import cythonize
import numpy
import os
import glob
import cdff_dev
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
        filenames = (glob.glob(cwd + "/cdff_*.cpp") +
                     glob.glob(cwd + "/cdff_*.so") +
                     glob.glob(cwd + "/cdff_*.pyd"))
        for filename in filenames:
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

    config.add_data_files(
        (".", "_cdff_types.pxd"),
        (".", "cdff_types.pxd"),
        (".", "cdff_types.pyx"),
        (".", "_cdff_envire.pxd"),
        (".", "cdff_envire.pxd"),
        (".", "cdff_envire.pyx")
    )

    extra_compile_args = [
        "-std=c++11",
        "-O3",
        # disable warnings caused by Cython using the deprecated
        # NumPy C-API
        "-Wno-cpp", "-Wno-unused-function"
    ]

    cdffpath = load_cdffpath()
    ctypespath = os.path.join(cdffpath, CTYPESDIR)

    cythonize("cdff_types.pyx")

    config.add_extension(
        "cdff_types",
        sources=["cdff_types.cpp"],
        include_dirs=[
            ".",
            numpy.get_include(),
            ctypespath
        ],
        library_dirs=[],
        libraries=[],
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )

    cythonize("cdff_envire.pyx")
    autoproj_current_root = os.environ.get("AUTOPROJ_CURRENT_ROOT", None)
    if autoproj_current_root is None:
        # TODO might be changed to warning?
        raise IOError("Environment variable $AUTOPROJ_CURRENT_ROOT is not "
                      "defined. Cannot build EnviRe bindings.")
    install_dir = os.path.join(autoproj_current_root, "install")

    config.add_extension(
        "cdff_envire",
        sources=["cdff_envire.cpp"],
        include_dirs=[
            ".",
            "envire",
            numpy.get_include(),
            os.path.join(install_dir, "include"),
            "/usr/local/include/eigen3",
            ctypespath
        ],
        library_dirs=[
            os.path.join(install_dir, "lib")
        ],
        libraries=["base-types", "envire_core", "envire_urdf", "urdfdom_model",
                   "envire_visualizer_interface"],
        define_macros=[("NDEBUG",)],
        extra_compile_args=extra_compile_args
    )

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    strict_prototypes_workaround()
    metadata = dict(
        name="cdff_dev",
        version=cdff_dev.__version__,
        description=cdff_dev.__description__,
        long_description=open("README.md").read(),
        scripts=["bin" + os.sep + "dfn_template_generator",
                 "bin" + os.sep + "dfpc_template_generator"],
        packages=['cdff_dev'],
        package_data={'cdff_dev': ['templates/*.template']},
        requires=['pyyaml', 'cython', 'Jinja2', 'numpy', 'pydot'],
        cmdclass = {'clean': CleanCommand},
        configuration=configuration
    )
    setup(**metadata)
