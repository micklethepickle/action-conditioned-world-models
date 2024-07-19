from flax import struct
from src.brax.custom_envs.myriad.base import IndirectFHCS

import jax.numpy as jnp
import jax
from typing import Dict, Any, Union


@struct.dataclass
class State:
    """Environment state for training and inference."""
    state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    timestep: int
    key: jnp.ndarray
    metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class MyriadEnv:
    "Euler's Integration"
    def __init__(self, cs_env: IndirectFHCS, T=20., dt=0.05, distractor_dims=0, noise=0.):
        """
        final state dim is always time step
        distractor_dims:
            Add n distractor dimensions with random noise sampled from N(0, 1).
        """
        self.cs_env = cs_env
        self.state_size = len(self.cs_env.x_0) 
        self.observation_size = self.state_size + 1 # plus timestep
        self.distractor_dims = distractor_dims
        self.noise = noise
        self.observation_size = self.observation_size + distractor_dims # distractor dimensions
        self.action_size = len(self.cs_env.bounds) - self.state_size
        
        self.T = T # Number of discrete timesteps. Not seconds
        self.dt = dt
        self.terminal_cost = cs_env.terminal_cost
        
    def unnormalize_action(self, action: Union[float, jnp.ndarray]):
        # actions are usually bounded by [-1, 1], but need to be bounded by self.cs_env.bounds
        lower_bounds = self.cs_env.bounds[-self.action_size:, 0] # (action_size,)
        upper_bounds = self.cs_env.bounds[-self.action_size:, 1] # (action_size,)
        midpoint = (lower_bounds + upper_bounds)/2.
        width = upper_bounds - lower_bounds
        
        return action * (width/2.) + midpoint
        
        
    def reset(self, key) -> State:
        cur_t = 0
        state = self.cs_env.x_0
        done, reward = jnp.zeros(2)
        
        init_state = State(state=state,
                          obs=jnp.append(state, cur_t), #  add timestep
                          reward=reward,
                          done=done,
                          timestep=cur_t,
                          key=key)
        return self.add_distractors(init_state)
    
    def make_reward_fn(self, batched=True):
        def get_reward(obs, action, key):
            u_t = self.unnormalize_action(action)
            x_t = obs[self.distractor_dims:-1]
            timestep = obs[-1]
            reward = -(self.cs_env.cost(x_t, u_t, self.dt*timestep)) * self.dt
            terminal_reward = jax.lax.cond(timestep == self.T - 1, 
                  lambda x, u: -self.cs_env.terminal_cost_fn(x, u),
                  lambda x, u: 0., 
                  x_t, u_t)
            return reward + terminal_reward
        
        if batched:
            return jax.vmap(get_reward, in_axes=(0, 0, None), out_axes=(0))
        else:
            return get_reward
        
    def get_initial_state(self, obs: jnp.ndarray, key: jnp.ndarray) -> State:
        done, reward = jnp.zeros(2)
        init_state = State(state=obs[self.distractor_dims:-1], # exclude timestep dim and distractors
                          obs=obs,
                          reward=reward,
                          done=done,
                          timestep=0,
                          key=key)
        
        return init_state
    
    def add_distractors(self, state: State):
        distractions = jax.random.normal(state.key, (self.distractor_dims, ))
        new_obs = jnp.append(distractions, state.obs)
        new_state = state.replace(obs=new_obs)
        return new_state
    
    def step(self, state: State, action: jnp.ndarray) -> State:
        u_t = self.unnormalize_action(action)
        x_t = state.state
        # print(self.dt * state.timestep)
        reward = jnp.squeeze(-(self.cs_env.cost(x_t, u_t, self.dt*state.timestep)) * self.dt)
        timestep = state.timestep
        terminal_reward = jax.lax.cond(timestep == self.T - 1, 
                              lambda x, u: -jnp.squeeze(self.cs_env.terminal_cost_fn(x, u)),
                              lambda x, u: 0., 
                              x_t, u_t)
        reward = reward + terminal_reward

        dx = self.cs_env.dynamics(x_t, u_t)
        next_state = x_t + (dx * self.dt)
        
        next_state = next_state + (jax.random.normal(state.key, next_state.shape) * self.noise)
        next_state = jnp.clip(next_state, self.cs_env.bounds[:self.state_size, 0], #lower bound
                                          self.cs_env.bounds[:self.state_size, 1]) #upper bound
        
        
        next_t = state.timestep + 1
        done = jax.lax.cond(next_t == self.T, lambda: 1., lambda : 0.)

        skey, _ = jax.random.split(state.key)
        new_state = state.replace(state=next_state,
                                 obs=jnp.append(next_state, next_t),
                                 reward=reward,
                                 done=done,
                                 timestep=next_t,
                                 key=skey)
        return self.add_distractors(new_state)