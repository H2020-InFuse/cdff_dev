def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration("cdff_dev", parent_package, top_path)
    config.add_subpackage("dfns")
    config.add_subpackage("dfpcs")
    config.add_subpackage("extensions")
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
