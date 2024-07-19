from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    
    config.alg = 'svginf'
    config.unroll_length = -1
    config.true_reward = True
    config.num_cycles = 10
    config.data_per_cycle = 50
    config.wm_steps = 20000
    config.policy_grad_tolerance = 100000
    config.max_policy_steps = 50
    config.reset_every = True
    config.batch_size = 64
    config.policy_batch_size = 8
    config.eval_every = 5
    config.action_repeat = 1
    config.network_sizes = (256, 256)
    config.bp_discount = 0.99
    config.discount = 1.
    config.bootstrap = 0
    config.entropy_init = 0.
    config.entropy_decay_rate = 1
    config.entropy_transition_steps = 100
    config.dynamics_lr = 0.001
    config.policy_lr = 0.001
    config.critic_lr = 0.001
    config.grad_clip = 10
    config.buffer_max = int(1e5)
    config.with_tqdm = False
    
    return config



