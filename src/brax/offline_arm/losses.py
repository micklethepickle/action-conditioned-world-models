from src.brax.offline_arm import networks as arm_networks
from brax.training.types import PRNGKey
from typing import Tuple, Any
import jax
import jax.numpy as jnp
from src.misc.helper_methods import detach, kl_mvn
import functools

batch_kl_mvn = jax.vmap(kl_mvn, in_axes=(0, 0, 0, 0), out_axes=(0))

def make_losses(arm_networks: arm_networks.ARMNetworks, 
				discount: float, bp_discount: float,
				 unroll_length: int,
				 make_policy,
				 reward_function,
				 policy_batch_size=16,
				 epsilon=1.,
				 bootstrap=1,
				 input_observations=False):
	transition_network = arm_networks.transition_network
	reward_network = arm_networks.reward_network
	policy_network = arm_networks.policy_network
	critic_network = arm_networks.critic_network

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

		# Reward loss. Epsilon decides between teacher forcing or student forcing. epsilon=1 is equivalent to only student forcing, epsilon=0 is only teacher forcing
		# mask = jax.random.uniform(key, shape=(observations.shape[0], observations.shape[1], 1)) < epsilon
		# interpolated_observations = (all_obs_predictions[:, :-1, :] * mask) + (observations * (1-mask))
		# reward_pred = reward_network.apply(preprocess_params, reward_params, detach(interpolated_observations), actions)
		# rerror = reward_pred - rewards
		# rloss = 0.5 * jnp.mean(jnp.square(rerror))

		return tloss, {'tloss': tloss, 'rloss': 0}

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
		obs, cache, key = carry # obs of shape (batch_size, 1, -1)
		key, transition_key, key_sample, key_reward = jax.random.split(key, 4)
		obs = detach(obs) + (bp_discount * obs) - (bp_discount * detach(obs))
		# bp_discounted_obs = detach(obs) + (bp_discount * obs) - (bp_discount * detach(obs))
		# bp_discounted_obs = obs
		action, extra = policy(obs, key_sample) # action of shape (batch_size, 1, -1)

		# get divergence from some other policy
		if other_policy is not None:
			_, other_extra = other_policy(obs, key_sample)
			divergence = batch_kl_mvn(extra['loc'], extra['scale'], other_extra['loc'], other_extra['scale'])
		else:
			divergence = jnp.zeros(policy_batch_size	)

		reward = reward_function(obs[:, 0], action, key_reward)
		# reward = reward_network.apply(preprocess_params, reward_params, obs, action) # reward of shape (batch_size, 1, 1)
		if input_observations:
			current_in = jnp.concatenate((obs, action), axis=-1)
		else:
			current_in = action
		next_obs, cache = transition_network.apply_recurrence(preprocess_params, transition_params, current_in,
																timesteps=jnp.array([tmp]),
																cache=cache, key=transition_key, train=False) # obs of shape (batch_size, 1, -1)


		return (next_obs, cache, key), (reward, obs, extra['entropy'], divergence)

	def batched_get_imagined_out(policy_params, preprocess_params, transition_params, reward_params, critic_params, 
								target_critic_params, init_obs, key,
								other_policy_params=None):
		# init_obs of shape (policy_batch_size, 1, -1)
		key, transition_key = jax.random.split(key)
		batch_size = policy_batch_size

		cache = transition_network.prime_recurrence(preprocess_params, transition_params,
													batch_size, unroll_length, init_obs, transition_key, train=False)
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
		(next_obs, _, _), (rewards, obs, entropy, divergence) = jax.lax.scan(f, (init_obs, cache, key), timesteps)
		rewards = jnp.transpose(jnp.squeeze(rewards), axes=(1, 0)) # rewards of shape (batch_size, length)
		trajectory_discounts = jnp.power(discount, jnp.arange(0, unroll_length))[jnp.newaxis, :]
		nstep_reward = jnp.sum(rewards * trajectory_discounts, axis=1) # (batch_size)

		bootstrapped = critic_network.apply(preprocess_params, critic_params, next_obs) # (batch_size, 1, 1)
		target_bootstrapped = critic_network.apply(preprocess_params, target_critic_params, next_obs)

		imagined_return = nstep_reward + (bootstrap*(discount ** unroll_length) * jnp.squeeze(bootstrapped))
		target_value = nstep_reward + (bootstrap*(discount ** unroll_length) * jnp.squeeze(target_bootstrapped))
		
		total_entropy = jnp.mean(jnp.squeeze(entropy), axis=0) # entropies summed over episode. End shape (batch_size)
		total_divergence = jnp.sum(jnp.squeeze(divergence), axis=0) # end shape (batch_size)

		return imagined_return, target_value, total_entropy, total_divergence

	def batch_policy_loss(policy_params, preprocess_params, transition_params, 
							reward_params, critic_params, target_critic_params,
							 init_obs, entropy_reg, key,
							 other_policy_params=None):
		# init obs of shape (policy_batch_size, 1, -1)
		imagined_return, target_value, total_entropies, total_divergence = batched_get_imagined_out(policy_params, preprocess_params,
															 transition_params, reward_params, critic_params, target_critic_params, init_obs, key,
															 other_policy_params=other_policy_params)
		averaged_returns = jnp.mean(imagined_return, axis=0)
		total_entropy = jnp.mean(total_entropies, axis=0)

		return -(averaged_returns + (entropy_reg*total_entropy)), {'target_value': target_value, 'entropy':total_entropy, 'img_ret': averaged_returns,
																	'divergence': jnp.mean(total_divergence, axis=0)}

	return dynamics_loss, batch_policy_loss, critic_loss