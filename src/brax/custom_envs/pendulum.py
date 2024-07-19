import jax
from jax import numpy as jnp
from flax import struct
from typing import Dict, Any

# @jax.jit
def _obs_to_th(obs):
	cos_th, sin_th, thdot = obs
	th = jnp.arctan2(sin_th, cos_th)
	return jnp.squeeze(jnp.array([th.reshape((1,)), thdot.reshape((1,))]))

# @jax.jit
def _th_to_obs(state):
	th, thdot = state
	cos_th, sin_th = jnp.cos(th), jnp.sin(th)
	next_state = jnp.array([cos_th, sin_th, thdot])
	return next_state

def angle_normalize(x):
	return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

@struct.dataclass
class State:
	"""Environment state for training and inference."""
	state: jnp.ndarray
	obs: jnp.ndarray
	reward: jnp.ndarray
	done: jnp.ndarray
	timestep: int
	metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
	info: Dict[str, Any] = struct.field(default_factory=dict)

class Pendulum:
	def __init__(self, T):
		self.max_speed = 8.
		self.observation_size = 3
		self.action_size = 1
		self.dt = 0.05
		self.g = 10.0
		self.m = 1.0
		self.l = 1.0
		self.action_cost = 0.001
		self.T = T


	def reset(self, key) -> State:
		state = jnp.array([0., 0.], dtype=jnp.float32)
		done, reward = jnp.zeros(2)
		init_state = State(state=state,
							obs=_th_to_obs(state),
							reward=reward,
							done=done,
							timestep=0)

		# print('true init state', init_state.state.shape, init_state.obs.shape)

		return init_state

	def make_reward_fn(self, batched=True):
		def get_reward(obs, action, key):
			action = action * 2
			obs = jnp.squeeze(obs)
			state = _obs_to_th(obs)
			th, thdot = state

			cost = jnp.sum(angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + self.action_cost * (action ** 2))
			reward = -cost

			return reward
		if batched:
			return jax.vmap(get_reward, in_axes=(0, 0, None), out_axes=(0))
		else:
			return get_reward

	def get_initial_state(self, obs: jnp.ndarray, key: jnp.ndarray):
		done, reward = jnp.zeros(2)
		init_state = State(state=_obs_to_th(obs),
							obs=obs,
							reward=reward,
							done=done,
							timestep=0)
		# print('made init state', init_state.state.shape, init_state.obs.shape)
		return init_state


	def step(self, state: State, action: jnp.ndarray) -> State:
		action = action[0] * 2 # Assume bounded actions between [-1, 1], 
								#so modify the bound for the purposes of this env. Max torque of 2
		th, thdot = state.state

		cost = jnp.sum(angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + self.action_cost * (action ** 2))
		reward = -cost

		newthdot = (thdot + (-3 * self.g / (2 * self.l) * jnp.sin(th + jnp.pi) + 3.0 / (
		self.m * self.l ** 2) * action) * self.dt)
		newth = th + newthdot * self.dt
		newthdot = jnp.clip(newthdot, -self.max_speed, self.max_speed)
		next_state = jnp.array([newth, newthdot])

		next_t = state.timestep + 1
		done = jax.lax.cond(next_t == self.T, lambda : 1., lambda : 0.)

		new_state = state.replace(state=next_state,
						obs=_th_to_obs(next_state),
						reward=reward,
						timestep=next_t,
						done=done)
		return new_state