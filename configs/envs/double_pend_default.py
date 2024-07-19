from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    config.env_type = 'DOUBLEPEND'
    config.length = 10
    config.dt = 0.05
    
    config.env_name = "{0}-{1}".format(config.env_type, config.length)

    return config

