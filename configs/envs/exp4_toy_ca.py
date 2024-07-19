from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()
    
    config.env_type = 'toy-ca'
    config.length = 20
    config.reward_std = 0.
    config.state_std = 0.
    config.distracted = True
    config.do_add_phase = False
    config.final_phase_ratio = 0.
    
    config.env_name = "{0}-{1}".format(config.env_type, config.length)

    return config

