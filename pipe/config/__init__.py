import os
import astropy.config as astropyconfig
from .pipeconf import get_conf_paths, cache_dir


class ConfigNamespace(astropyconfig.ConfigNamespace):
    rootname = 'pipe-cheops'


class ConfigItem(astropyconfig.ConfigItem):
    rootname = 'pipe-cheops'


DATA_ROOT, REF_LIB_PATH = get_conf_paths()


class Conf(ConfigNamespace):
    """
    Configuration parameters for my subpackage.
    """
    ref_lib_data = ConfigItem(REF_LIB_PATH, 'Path to the reference files')
    data_root = ConfigItem(DATA_ROOT, 'Path to data files')

    for config_dir in [REF_LIB_PATH, DATA_ROOT]:
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)


conf = Conf()
