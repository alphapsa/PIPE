import json
import os
import astropy.config as astropyconfig

def cache_dir():
    return astropyconfig.get_cache_dir()


def get_conf_paths(overwrite=False):
    # Get configuration information from setup.cfg
    conf_path = os.path.join(os.path.dirname(__file__), 'conf.json')
    
    if (not os.path.isfile(conf_path))|(overwrite):
        # First creating a file
        data_path = input('Please enter a path to the data files\nPress ENTER to use the home directory\n(a directory called cheops-pipe will be created inside the home directory): ')
        ref_path = input('Please enter a path to the calibration files\nPress ENTER to use the home directory\n(a directory called cheops-pipe will be created inside the home directory): ')
        if data_path == '':
            data_path = os.path.expanduser('~') + '/cheops-pipe/Data'
        if ref_path == '':
            ref_path = os.path.expanduser('~') + '/cheops-pipe/Ref'
        fconf = open(os.path.dirname(__file__) + '/conf.json', 'w')
        fconf.write('{\n')
        fconf.write('\t"data_root" : "' + data_path + '",\n')
        fconf.write('\t"ref_lib_data" : "' + ref_path + '"\n')
        fconf.write('}')
        fconf.close()
        # And loading it
        conf_path = os.path.join(os.path.dirname(__file__), 'conf.json')

    with open(conf_path, 'r') as confstream:
        confparse = json.load(confstream)
    data_root_cfg = confparse.get("data_root")
    ref_lib_cfg = confparse.get("ref_lib_data")

    if len(data_root_cfg) == 0 and len(ref_lib_cfg) == 0:
        cache_dir = os.path.join(astropyconfig.get_cache_dir(), '.pipe-cheops')
        DATA_ROOT = os.path.join(cache_dir, 'data_root')
        REF_LIB_PATH = os.path.join(cache_dir, 'ref_lib_data')
    else:
        DATA_ROOT = data_root_cfg
        REF_LIB_PATH = ref_lib_cfg

    return DATA_ROOT, REF_LIB_PATH