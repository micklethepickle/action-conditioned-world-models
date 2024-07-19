from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    
    config.alg = 'svg'
    config.unroll_length = -1
    config.num_steps = 5000
    config.true_reward = True
    config.difference_transition = True
    config.batch_size = 64
    config.policy_batch_size = 16
    config.action_repeat = 1
    config.network_sizes = (64, 64)
    config.bp_discount = 0.9
    config.discount = 1.
    config.bootstrap = 0.
    config.entropy_init = 0.1
    config.entropy_decay_rate = 1.
    config.entropy_transition_steps = 100
    config.dynamics_lr = 0.001
    config.policy_lr = 0.
    config.critic_lr = 0.001
    config.tau = 0.005
    config.grad_clip = 100
    config.buffer_max = int(1e5)
    config.with_tqdm = False
    
    return config



