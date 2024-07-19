import jax
from jax import numpy as jnp
from flax import struct
from typing import Dict, Any


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

class Cancer:
	def __init__(self, r=0.3, a=3., delta=0.45, x_0=0.975, T=20):
		# Starting state
		self.T = T  # Duration of experiment
		# Bounds over the states (x_0, x_1 ...) are given first,
		# followed by bounds over controls (u_0, u_1,...)

		self.observation_size = 2
		self.action_size = 1

		self.adj_T = None  # Final condition over the adjoint, if any
		self.grow_rate = r
		self.cost_weight = a  # Positive weight parameter
		self.dose = delta  # Magnitude of the dose administered
		self.dt = 1 / T
		self.x_0 = x_0


	def reset(self, key) -> State:
		state = jnp.array([self.x_0], dtype=jnp.float32) # starting state
		done, reward = jnp.zeros(2)
		init_state = State(state=state,
							obs=jnp.append(state, 0),
							reward=reward,
							done=done,
							timestep=0)

		return init_state

	def make_reward_fn(self, batched=True):
		def get_reward(obs, action, key):
			u_t = action + 1.
			x_t, _ = obs
			d_x = self.grow_rate * x_t * jnp.log(1 / x_t) - u_t * self.dose * x_t
			next_state = x_t + d_x * self.dt
			reward = -(self.cost_weight * x_t ** 2 + u_t ** 2).sum()

			return reward
		if batched:
			return jax.vmap(get_reward, in_axes=(0, 0, None), out_axes=(0))
		else:
			return get_reward

	def get_initial_state(self, obs: jnp.ndarray, key: jnp.ndarray):
		done, reward = jnp.zeros(2)
		init_state = State(state=obs[0],
							obs=obs,
							reward=reward,
							done=done,
							timestep=0)
		return init_state

	def step(self, state: State, action: jnp.ndarray) -> State:
		u_t = action[0] + 1. # action bounded at [0., 2.]
		x_t = state.state
		d_x = self.grow_rate * x_t * jnp.log(1 / x_t) - u_t * self.dose * x_t
		next_state = x_t + d_x * self.dt
		reward = -(self.cost_weight * x_t ** 2 + u_t ** 2).sum()
		next_state = jnp.clip(next_state, 1e-3, 1.) # state bound

		next_t = state.timestep + 1
		done = jax.lax.cond(next_t == self.T, lambda : 1., lambda : 0.)
		new_state = state.replace(state=next_state,
									obs=jnp.append(next_state, next_t),
									reward=reward,
									done=done,
									timestep=next_t)

		return new_state