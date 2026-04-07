from cdff_dev.dfns.setup import get_extensions as get_dfns_extensions
from cdff_dev.dfpcs.setup import get_extensions as get_dfpcs_extensions
from cdff_dev.extensions.setup import get_extensions as get_ext_extensions


def get_extensions():
    return get_dfns_extensions() + get_dfpcs_extensions() + get_ext_extensions()