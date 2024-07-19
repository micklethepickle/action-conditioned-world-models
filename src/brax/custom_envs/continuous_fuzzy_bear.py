import jax.numpy as jnp
import jax
from flax import struct
from typing import Dict, Any
import numpy as np

@struct.dataclass
class State:
    """Environment state for training and inference."""
    state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    key: jnp.ndarray
    metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

class ContinuousFuzzyBear:
    def __init__(self, std=1.):
        self.std = std
        self.dt = 1.
        self.length = 2
        self.observation_size = 2
        self.action_size = 1
    
    def get_initial_state(self, obs: jnp.ndarray, key: jnp.ndarray):
        done, reward = jnp.zeros(2)
        init_state = State(state=obs,
                           obs=obs,
                           reward=reward,
                           done=done,
                           key=key)
        return init_state

    def reset(self, rng: jnp.ndarray):
        state = jnp.array([0., 0.]) # bear_type, timestep
        done, reward = jnp.zeros(2)
        init_state = State(state=state,
                           obs=state,
                           reward=reward,
                           done=done,
                           key=rng)
        return init_state

    def make_reward_fn(self, batched=True):
        def get_reward(obs, action, key):
            bear, timestep = obs[0], obs[-1]
            timestep = jnp.round(timestep)
            reward = jax.lax.cond(timestep == 2,
                                  lambda x: x,
                                  lambda x: 0.,
                                  bear)
            return reward

        if batched:
            return jax.vmap(get_reward, in_axes=(0,0,None), out_axes=(0))
        else:
            return get_reward
    
    def get_true_noise(self, obs, batch_size, key):
        # of size (batch_size, 2)
        noise = jax.lax.cond(jnp.all(jnp.isclose(obs[:, -1], 1.)),
                             lambda : jnp.concatenate((jax.random.normal(key, (batch_size, 1, 1)) * self.std,
                                                        jnp.zeros((batch_size, 1, 1))), axis=-1),
                             lambda: jnp.zeros((batch_size, 1, 2)))
        
        return noise

    def get_reward(self, state, action, key):
        bear, timestep = state.state[0], state.state[-1]
        reward = jax.lax.cond(timestep == 2,
                              lambda x: x,
                              lambda x: 0.,
                              bear)
        return reward

    def step(self, state, action):
        bear, timestep = state.state[0], state.state[-1]
        
        state_key, reward_key, key = jax.random.split(state.key, 3)

        reward = self.get_reward(state, action, reward_key)
        done = jax.lax.cond(timestep == 2, lambda : 1., lambda : 0.)

        next_bear = jax.lax.cond(timestep == 0, 
                                  lambda x, a: jnp.squeeze(jax.random.normal(state_key) * self.std),
                                  lambda x, a: jnp.squeeze(x * a),
                                  bear, action)
        
        next_state = jnp.array([next_bear, timestep + 1])
        new_state = state.replace(state=next_state,
                                obs=next_state,
                                reward=reward,
                                done=done,
                                key=key)
        return new_state