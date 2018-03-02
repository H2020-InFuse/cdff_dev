#!/usr/bin/env python3
from distutils.extension import Extension
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize
import numpy
import os
import cdff_dev
from cdff_dev.path import check_cdffpath, CTYPESDIR


def strict_prototypes_workaround():
    # Workaround to remove '-Wstrict-prototypes' from compiler invocation
    opt = get_config_vars('OPT')[0]
    os.environ['OPT'] = " ".join(flag for flag in opt.split()
                                 if flag != '-Wstrict-prototypes')


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
        (".", "cdff_types.pyx")
    )

    cdffpath = "CDFF"  # TODO
    check_cdffpath(cdffpath)
    ctypespath = os.path.join(cdffpath, CTYPESDIR)

    pyx_filename = os.path.join("cdff_types.pyx")
    cythonize(pyx_filename)

    config.add_extension(
        "cdff_types",
        sources=["cdff_types.cpp"],
        include_dirs=[
            ".",
            numpy.get_include(),
            ctypespath
        ],
        library_dirs=[
            ctypespath
        ],
        libraries=[],
        define_macros=[("NDEBUG",)],
        extra_compile_args=[
            "-std=c++11",
            "-O3",
            # disable warnings caused by Cython using the deprecated
            # NumPy C-API
            "-Wno-cpp", "-Wno-unused-function"
        ]
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
        scripts=["bin" + os.sep + "dfn_template_generator"],
        packages=['cdff_dev'],
        package_data={'cdff_dev': ['templates/*.template']},
        requires=['pyyaml', 'cython', 'Jinja2'],
        configuration=configuration
    )
    setup(**metadata)

