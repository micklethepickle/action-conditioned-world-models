from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    
    config.alg = 'svg'
    config.unroll_length = 2
    config.num_steps = 5000
    config.warmup_steps = 1500
    config.dynamics_update_every = 2
    config.policy_update_every = 16
    config.true_reward = True
    config.difference_transition = True
    config.batch_size = 64
    config.policy_batch_size = 16
    config.eval_every = -1
    config.action_repeat = 1
    config.network_sizes = (64, 64)
    config.bp_discount = 0.9
    config.discount = 1.
    config.bootstrap = 1.
    config.entropy_init = 0.1
    config.entropy_decay_rate = 1.
    config.entropy_transition_steps = 100
    config.dynamics_lr = 0.001
    config.policy_lr = 0.0001
    config.critic_lr = 0.001
    config.tau = 0.005
    config.grad_clip = 100
    config.buffer_max = int(1e6)
    config.with_tqdm = False
    
    return config



