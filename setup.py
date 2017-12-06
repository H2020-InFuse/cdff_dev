#!/usr/bin/env python3
from distutils.core import setup
import os
import cdff_dev


if __name__ == "__main__":
    setup(name='cdff_dev',
          version=cdff_dev.__version__,
          description=cdff_dev.__description__,
          long_description=open("README.md").read(),
          scripts=["bin" + os.sep + "dfn_template_generator"],
          packages=['cdff_dev'],
          package_data={'cdff_dev': ['templates/*.template']},
          requires=['pyyaml', 'cython', 'Jinja2'])

