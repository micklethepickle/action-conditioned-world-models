from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    config.env_type = 'CANCER'
    config.length = 100
    config.constant_dt = False
    config.distractor_dims = 0
    config.noise = 0.
    
    config.env_name = "{0}-{1}".format(config.env_type, config.length)

    return config

