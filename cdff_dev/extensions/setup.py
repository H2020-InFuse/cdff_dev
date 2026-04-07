from cdff_dev.extensions.gps.setup import get_extensions as get_gps_extensions
from cdff_dev.extensions.pcl.setup import get_extensions as get_pcl_extensions


def get_extensions():
    return get_gps_extensions() + get_pcl_extensions()