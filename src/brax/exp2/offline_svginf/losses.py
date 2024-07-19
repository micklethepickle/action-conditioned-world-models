from src.brax.exp2.offline_svginf import networks as svg_networks
from brax.training.types import PRNGKey
from typing import Tuple, Any
import jax
import jax.numpy as jnp
from src.misc.helper_methods import detach, kl_mvn
from brax import envs
import functools

batch_kl_mvn = jax.vmap(kl_mvn, in_axes=(0, 0, 0, 0), out_axes=(0))

def make_losses(svg_networks: svg_networks.SVGNetworks, discount: float, bp_discount: float,
				 env: envs.Env, unroll_length: int,
				 make_policy,
				 reward_function=None,
				 policy_batch_size=16,
				 bootstrap=1):
	transition_network = svg_networks.transition_network
	reward_network = svg_networks.reward_network
	policy_network = svg_networks.policy_network
	critic_network = svg_networks.critic_network

	def transition_loss(transition_params, preprocess_params, observations, actions, next_observations):
		next_obs_predictions = transition_network.apply(preprocess_params, transition_params, observations, actions)
		error = next_observations - next_obs_predictions
		loss = 0.5 * jnp.mean(jnp.square(error))
		return loss

	def reward_loss(reward_params, preprocess_params, observations, actions, rewards):
		if reward_function is not None:
			return 0
		else:
			reward_pred = reward_network.apply(preprocess_params, reward_params, observations, actions)
			error = reward_pred - rewards
			loss = 0.5 * jnp.mean(jnp.square(error))

			return loss

	def critic_loss(critic_params, preprocess_params, observations, target_value, key):
		"""
		TODO: Decide on what target_value should be. TD error? Dreamer uses lambda returns on imagined rewards. 
			We don't use lambda returns for policy gradient though. To be consistent, should use n-step return,
			same as policy gradient.
		"""

		value_prediction = critic_network.apply(preprocess_params, critic_params, detach(observations))
		loss = 0.5 * jnp.mean(jnp.square(target_value - jnp.squeeze(value_prediction)))

		return loss


	def step(carry: Tuple[Any, PRNGKey], tmp, policy, preprocess_params, transition_params, reward_params, other_policy=None):
		obs, key = carry
		key, key_sample, key_reward = jax.random.split(key, 3)
		action, extra = policy(obs, key_sample)
		if reward_function is not None:
			reward = reward_function(obs[:, 0], action, key_reward)
		else:
			reward = reward_network.apply(preprocess_params, reward_params, obs, action)

		# get divergence from some other policy
		if other_policy is not None:
			_, other_extra = other_policy(obs, key_sample)
			divergence = batch_kl_mvn(extra['loc'], extra['scale'], other_extra['loc'], other_extra['scale'])
		else:
			divergence = jnp.zeros(policy_batch_size)

		next_obs = transition_network.apply(preprocess_params, transition_params, obs, action)
		next_obs = detach(next_obs) + (bp_discount * next_obs) - (bp_discount * detach(next_obs))

		return (next_obs, key), (reward, obs, extra['entropy'], divergence)

	def batched_get_imagined_out(policy_params, preprocess_params, transition_params, reward_params, critic_params, init_obs, key, other_policy_params=None):
		# init_obs of shape (policy_batch_size, 1, -1)
		key, transition_key = jax.random.split(key)
		batch_size = policy_batch_size

		timesteps = jnp.arange(1, unroll_length + 1)

		if other_policy_params is not None:
			other_policy = make_policy((preprocess_params, other_policy_params),  get_dist=True)
		else:
			other_policy = None

		f = functools.partial(step, 
							policy=make_policy((preprocess_params, policy_params), get_dist=True),
							preprocess_params=preprocess_params,
							transition_params=transition_params,
							reward_params=reward_params,
							other_policy=other_policy)
		(next_obs, _), (rewards, obs, entropy, divergence) = jax.lax.scan(f, (init_obs, key), None, unroll_length)
		rewards = jnp.transpose(jnp.squeeze(rewards), axes=(1, 0)) # rewards of shape (batch_size, length)
		trajectory_discounts = jnp.power(discount, jnp.arange(0, unroll_length))[jnp.newaxis, :]
		nstep_reward = jnp.sum(rewards * trajectory_discounts, axis=1) # (batch_size)
		bootstrapped = critic_network.apply(preprocess_params, critic_params, next_obs) # (batch_size, 1, 1)
		target_value = nstep_reward + (bootstrap*(discount ** unroll_length) * jnp.squeeze(bootstrapped))

		total_entropy = jnp.sum(jnp.squeeze(entropy), axis=0) # entropies summed over episode. End shape (batch_size)
		total_divergence = jnp.sum(jnp.squeeze(divergence), axis=0) # end shape (batch_size)

		return target_value, total_entropy, total_divergence

	def batch_policy_loss(policy_params, preprocess_params, transition_params, reward_params, 
							critic_params, init_obs, entropy_reg, key,
							other_policy_params=None):
		# all_keys = jax.random.split(key, policy_batch_size)
		target_value, total_entropies, total_divergence = batched_get_imagined_out(policy_params, preprocess_params, transition_params, 
																reward_params, critic_params, init_obs, key,
																other_policy_params=other_policy_params)
		averaged_values = jnp.mean(target_value, axis=0)
		total_entropy = jnp.mean(total_entropies, axis=0)

		return -(averaged_values + (entropy_reg*total_entropy)), {'img_rew': target_value, 'entropy':total_entropy, 'img_ret': averaged_values,
																	'divergence': jnp.mean(total_divergence, axis=0)}

	return transition_loss, reward_loss, batch_policy_loss, critic_loss