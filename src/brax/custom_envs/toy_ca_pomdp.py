import jax.numpy as jnp
import jax
from flax import struct
from typing import Dict, Any
import numpy as np

GOAL_INITIAL_ACTION  = 0.5

def final_reward(x):
  return jnp.clip(-20 * (x - GOAL_INITIAL_ACTION) **2 + 10, a_min=-10)

def short_term_reward(key, rstate, action, timestep, length, std):
  key, skey = jax.random.split(key)
  reward = jax.lax.cond(jnp.round(timestep) == 0,
                        lambda: 0.,
                        lambda: (-((rstate - action)**2)/length) + jax.random.normal(skey) * std)
  return reward

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

class ToyCA:
  def __init__(self, length, reward_std=1., state_std=1., do_add_phase=True, is_distracted=True, final_phase_ratio=0):
    if do_add_phase:
      self.observation_size = 5
      self.add_timestep_embd = self.add_phase
    else:
      self.observation_size = 3
      self.add_timestep_embd = self.state_identity
    self.action_size = 1
    self.length = length
    self.reward_std = reward_std
    self.state_std = state_std
    self.dt = 1
    self.is_distracted = is_distracted
    if final_phase_ratio == 0:
        self.final_phase_length = 1
    else:
        self.final_phase_length = int(length * final_phase_ratio)
    # self.get_pe = make_get_pe(positional_size)
  
  def get_initial_state(self, obs: jnp.ndarray, key: jnp.ndarray):
    done, reward = jnp.zeros(2)
    init_state = State(state=obs,
                      obs=obs,
                      reward=reward,
                      done=done,
                      key=key)
    return init_state

  def state_identity(self, state):
    return state
  def add_phase(self, state):
    _, _, timestep = state
    phase = jax.lax.cond(timestep >= self.length - self.final_phase_length,
                        lambda: jnp.array([1, 0]),
                        lambda: jnp.array([0, 1]))
    return jnp.concatenate((state, phase))

  def reset(self, rng: jnp.ndarray):
    state = jnp.array([0., 0., 0.]) # saved_action, rstate, timestep
    state = self.add_timestep_embd(state)
    done, reward = jnp.zeros(2)
    init_state = State(state=state,
              obs=state,
              reward=reward,
              done=done,
              key=rng)
    return init_state

  def get_random_action(self, key):
    skey, key = jax.random.split(key)
    action = jax.random.uniform(akey, minval=-1., maxval=1.)
    return action
  
  def make_reward_fn(self, batched=True):
    def get_reward(obs, action, key):
      sa, rstate, timestep = obs[0], obs[1], obs[-1]
      timestep = jnp.round(timestep)
      reward = jax.lax.cond(timestep >= self.length - self.final_phase_length, 
                  lambda x, y, a: final_reward(x)/self.final_phase_length,
                  lambda x, y, a: short_term_reward(key, y, a, timestep, self.length, self.reward_std) * self.is_distracted, 
                  sa, rstate, jnp.squeeze(action))

      return reward
    if batched:
      return jax.vmap(get_reward, in_axes=(0, 0, None), out_axes=(0))
    else:
      return get_reward


  def get_reward(self, state, action, key):
    sa, rstate, timestep = state.state[0], state.state[1], state.state[-1]
    reward = jax.lax.cond(timestep >= self.length - self.final_phase_length, 
                lambda x, y, a: final_reward(x)/self.final_phase_length,
                lambda x, y, a: short_term_reward(key, y, a, timestep, self.length, self.reward_std) * self.is_distracted, 
                sa, rstate, jnp.squeeze(action))
    
    return reward

  def get_optimal_action(self, state):
    sa, rstate, timestep = state.state[0], state.state[1], state.state[-1]
    action = jax.lax.cond(timestep == 0,
                          lambda s: GOAL_INITIAL_ACTION,
                          lambda s: jnp.clip(rstate, a_min=-1., a_max=1.),
                          rstate)
    return action

  def get_short_term_optimal_action(self, state, key):
    sa, rstate, timestep = state.state[0], state.state[1], state.state[-1]
    akey, key = jax.random.split(key)
    action = jax.lax.cond(timestep == 0,
                          lambda s: jax.random.uniform(akey, minval=-1., maxval=1.),
                          lambda s: jnp.clip(rstate, a_min=-1., a_max=1.),
                          rstate)
    return action

  
  def step(self, state, action):
    sa, rstate, timestep = state.state[0], state.state[1], state.state[-1]
    
    rstate_key, reward_key, key = jax.random.split(state.key, 3)
    new_rstate = jax.random.normal(rstate_key) * self.state_std
    new_timestep = timestep + 1

    new_sa = jax.lax.cond(timestep == 0, lambda x, y: x, lambda x, y: y,
                jnp.squeeze(action), sa)
    reward = self.get_reward(state, action, reward_key)
    done = jax.lax.cond(timestep == self.length - 1, lambda : 1., lambda : 0.)

    next_state = jnp.array([new_sa, new_rstate, new_timestep])
    next_state = self.add_timestep_embd(next_state)
    new_state = state.replace(state=next_state,
                 obs=next_state,
                 reward=reward,
                 done=done,
                 key=key)
    return new_state