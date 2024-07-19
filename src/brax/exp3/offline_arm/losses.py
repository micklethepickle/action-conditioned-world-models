from src.brax.exp3.offline_arm import networks as arm_networks
from brax.training.types import PRNGKey
from typing import Tuple, Any
import jax
import jax.numpy as jnp
from src.misc.helper_methods import detach, kl_mvn
import functools

batch_kl_mvn = jax.vmap(kl_mvn, in_axes=(0, 0, 0, 0), out_axes=(0))

def make_losses(arm_networks: arm_networks.ARMNetworks, 
				 input_observations=False):
	transition_network = arm_networks.transition_network
	reward_network = arm_networks.reward_network

	def dynamics_loss(dynamics_params, preprocess_params, observations, actions, rewards, next_observations, key):
		# actions of size (B, L, A)
		key, transition_key = jax.random.split(key)
		batch_size, length, action_dim = actions.shape
		transition_params, reward_params = dynamics_params

		# Dynamics loss
		if input_observations:
			history = jnp.concatenate((observations, actions), axis=-1)
		else:
			history = actions
		init_obs = observations[:, 0:1, :]
		all_obs_predictions = transition_network.apply_sequence(preprocess_params, transition_params, init_obs, history, key=transition_key)
		next_obs_predictions = all_obs_predictions[:, 1:, :] # exclude the initial observation prediction
		terror = next_observations - next_obs_predictions
		tloss = 0.5 * jnp.mean(jnp.square(terror))

		total_loss = tloss

		return total_loss, {'tloss': tloss}

	return dynamics_loss