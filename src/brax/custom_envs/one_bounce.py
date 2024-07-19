import jax
from jax import numpy as jnp
from flax import struct
from typing import Dict, Any
from jax.numpy import sin, cos


@struct.dataclass
class State:
    """Environment state for training and inference."""
    state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

class OneBounce:
    def __init__(self, goal_state, length, dt=0.01, friction_constant=10., wall_position=-1.):
        self.observation_size = 5 # ball position, ball velocity, wall position, goal_position, timestep
        self.action_size = 1
        self.dt = dt
        self.goal_state = goal_state
        self.length = length
        self.friction_constant = friction_constant
        self.wall_position = wall_position
    
    def reset(self, key) -> State:
        state = jnp.array([0., 0., self.wall_position, self.goal_state, 0.])
        done, reward = jnp.zeros(2)
        init_state = State(state=state,
                          obs=state, 
                          reward=reward,
                          done=done)
        
        return init_state
    
    def step(self, state: State, action: jnp.ndarray) -> State:
        action = (action - 2) * 10
        
        ball_x, ball_v, wall_x, goal_x, t = state.state
        reward = jax.lax.cond(t == self.length - 1,
                             lambda bx, gx: -(bx - gx)**2,
                             lambda bx, gx: 0.,
                             ball_x, goal_x)
        
        delta_x = (ball_v * self.dt)
        delta_v = jax.lax.cond(ball_x + delta_x <= self.wall_position,
                      lambda v: -2*v,
                      lambda v: (-self.friction_constant * v) * self.dt,
                      ball_v)
        delta_x = jax.lax.cond((ball_x + delta_x) <= self.wall_position,
                              lambda x: 2*(self.wall_position - x) - delta_x,
                              lambda x: delta_x,
                              ball_x)


        delta = jnp.array([delta_x, delta_v, 0., 0., 1.])
        
        next_s = jax.lax.cond(t == 0,
                             lambda s: jnp.array([0., action, self.wall_position, self.goal_state, 1.]),
                             lambda s: s + delta,
                             state.state)
        
        done = jax.lax.cond(t == self.length - 1, lambda: 1., lambda: 0.)
        
        new_state = state.replace(state=next_s,
                                  reward=reward,
                                  done=done,
                                  obs=next_s)
        return new_state

    def get_short_term_optimal_action(self, state, key):
        timestep = state.state[-1]
        akey, key = jax.random.split(key)
        action = jax.lax.cond(timestep == 0,
                              lambda k: jax.random.uniform(k, shape=(self.action_size,), minval=-1., maxval=1.),
                              lambda k: jnp.zeros(self.action_size),
                              akey)
        return action

    def make_reward_fn(self, batched=True):
        def get_reward(obs, action, key):
            ball_x, ball_v, wall_x, goal_x, t = obs
            t = jnp.round(t)
            
            reward = jax.lax.cond(t == self.length - 1,
                                 lambda bx, gx: -(bx - gx)**2,
                                 lambda bx, gx: 0.,
                                 ball_x, goal_x)

            return reward
        if batched:
            return jax.vmap(get_reward, in_axes=(0, 0, None), out_axes=(0))
        else:
            return get_reward

    def get_initial_state(self, obs: jnp.ndarray, key: jnp.ndarray):
        done, reward = jnp.zeros(2)
        init_state = State(state=obs,
                            obs=obs,
                            reward=reward,
                            done=done)
        return init_state