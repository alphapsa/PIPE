import os
import astropy.config as astropyconfig

class ConfigNamespace(astropyconfig.ConfigNamespace):
    rootname = 'pipe-cheops'


class ConfigItem(astropyconfig.ConfigItem):
    rootname = 'pipe-cheops'


cache_dir = os.path.join(astropyconfig.get_cache_dir(), '.pipe-cheops')
REF_LIB_PATH = os.path.join(cache_dir, 'ref_lib_data')
DATA_ROOT = os.path.join(cache_dir, 'data_root')


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
