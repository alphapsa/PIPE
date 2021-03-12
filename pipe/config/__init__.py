import os
import astropy.config as astropyconfig

# Get configuration information from setup.cfg
from configparser import ConfigParser
conf = ConfigParser()

conf.read([os.path.join(os.path.dirname(__file__), '..', 'setup.cfg')])
default_paths = dict(conf.items('default_paths'))


class ConfigNamespace(astropyconfig.ConfigNamespace):
    rootname = 'pipe-cheops'


class ConfigItem(astropyconfig.ConfigItem):
    rootname = 'pipe-cheops'


if (len(default_paths['data_root']) == 0 and
        len(default_paths['ref_lib_data']) == 0):
    cache_dir = os.path.join(astropyconfig.get_cache_dir(), '.pipe-cheops')
    REF_LIB_PATH = os.path.join(cache_dir, 'ref_lib_data')
    DATA_ROOT = os.path.join(cache_dir, 'data_root')
else:
    DATA_ROOT = default_paths['data_root']
    REF_LIB_PATH = default_paths['ref_lib_data']


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
