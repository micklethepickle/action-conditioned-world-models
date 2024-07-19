from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    config.env_type = 'double_pend'
    config.length = 20
    
    config.env_name = "{0}-{1}".format(config.env_type, config.length)

    return config

