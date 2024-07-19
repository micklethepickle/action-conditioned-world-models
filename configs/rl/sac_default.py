from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    
    config.alg = 'sac'
    config.num_steps = 5000
    config.warmup_steps = 1000
    config.dynamics_update_every = 2
    config.policy_update_every = 4
    config.batch_size = 128
    config.eval_every = -1
    config.action_repeat = 1
    config.network_sizes = (128, 128)
    config.discount = 0.99
    config.entropy_init = 0.1
    config.lr = 3e-4
    config.tau = 0.005
    config.buffer_max = int(1e6)
    config.with_tqdm = False
    
    return config



