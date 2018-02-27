#!/usr/bin/env python3
import os
import cdff_dev


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
        (".", "cdff_dev/cdff_types/_cdff_types.pxd"),
        (".", "cdff_dev/cdff_types/cdff_types.pxd"),
        (".", "cdff_dev/cdff_types/cdff_types.pyx")
    )

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

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

