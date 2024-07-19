import jax
from jax import numpy as jnp
from flax import struct
from typing import Dict, Any
from jax.numpy import sin, cos
from jax.experimental.ode import odeint


G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


@jax.jit
def derivs(state, t):
    dydx0 = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx1 = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx2 = state[3]

    den2 = (L2/L1) * den1
    dydx3 = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return jnp.array([dydx0, dydx1, dydx2, dydx3])

def get_feasible_goal(length, dt, key, goal_action=None):
    if goal_action is None:
        goal_action = jax.random.uniform(key, shape=(2,), minval=-180, maxval=180)

    initial_state = jnp.array([goal_action[0], 0., goal_action[1], 0.])
    y = initial_state
    t = 0
    for i in range(length - 2):
        # y = y + derivs(y) * dt
        y = odeint(derivs, y, jnp.array([i* dt,(i*dt) + dt]))[-1]
        # print(y)
    return y, goal_action

def get_goal_state(length, dt, goal_action):
    initial_state = jnp.array([goal_action[0], 0., goal_action[1], 0.])
    y = initial_state
    t = 0
    for i in range(length - 2):
        # y = y + derivs(y) * dt
        y = odeint(derivs, y, jnp.array([i* dt,(i*dt) + dt]))[-1]
        # print(y)
    return y, jnp.array(goal_action)

@struct.dataclass
class State:
    """Environment state for training and inference."""
    state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

class DoublePendulum:
    def __init__(self, goal_state, goal_action, length, dt=0.01):
        self.observation_size = 5 # position1, velocity1, position2, velocity2, timestep
        self.action_size = 1
        self.dt = dt
        self.goal_state = goal_state
        self.goal_action = goal_action
        self.length = length

    def reset(self, key) -> State:
        state = jnp.array(self.goal_state, dtype=jnp.float32)
        state = jnp.append(state, 0)
        done, reward = jnp.zeros(2)
        init_state = State(state=state,
                            obs=state,
                            reward=reward,
                            done=done)

        return init_state

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
            timestep = obs[-1]
            timestep = jnp.round(timestep)
            
            reward = jax.lax.cond(timestep == self.length - 1,
                                 lambda s: -jnp.mean((s - self.goal_state)**2),
                                 lambda s: 0.,
                                 obs[:-1])

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


    def step(self, state: State, action: jnp.ndarray) -> State:
        action = action * 180 # Assume bounded actions between [-1, 1], 
                                #so modify the bound for the purposes of this env. Max torque angles [-180, 180]
                                # action is to set the initial angles of balls
                
        timestep = state.state[-1]

        next_s = jax.lax.cond(timestep == 0,    
                             lambda s, a: jnp.array([jnp.squeeze(action), 0., self.goal_action[1]*180., 0.]),
                             # lambda s, a: s + derivs(s) * self.dt,
                             lambda s, a: odeint(derivs, s, jnp.array([timestep * self.dt,(timestep*self.dt) + self.dt]))[-1],
                             state.state[:-1], action)
        next_s = jnp.append(next_s, timestep + 1)
        
        reward = jax.lax.cond(timestep == self.length - 1,
                             lambda s: -jnp.mean((s - self.goal_state)**2),
                             lambda s: 0.,
                             state.state[:-1])
        done = jax.lax.cond(timestep == self.length - 1, lambda: 1., lambda: 0.)

        new_state = state.replace(state=next_s,
                                  reward=reward,
                                  done=done,
                                  obs=next_s)
        return new_state
