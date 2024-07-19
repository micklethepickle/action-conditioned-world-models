from src.brax.exp3.offline_svginf import networks as svg_networks
from brax.training.types import PRNGKey
from typing import Tuple, Any
import jax
import jax.numpy as jnp
from src.misc.helper_methods import detach, kl_mvn
from brax import envs
import functools

def make_losses(svg_networks: svg_networks.SVGNetworks,
				 env: envs.Env):
	transition_network = svg_networks.transition_network

	def transition_loss(transition_params, preprocess_params, observations, actions, next_observations):
		# IMPORTANT, USE DIFFERENCE PREDICTION
		next_obs_predictions = transition_network.apply(preprocess_params, transition_params, observations, actions)
		error = next_observations - next_obs_predictions
		loss = 0.5 * jnp.mean(jnp.square(error))
		return loss


	return transition_loss