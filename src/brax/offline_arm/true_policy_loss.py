from src.brax.offline_arm import networks as arm_networks
from brax.training.types import PRNGKey
from typing import Tuple, Any
import jax
import jax.numpy as jnp
from src.misc.helper_methods import detach
import functools

def make_loss(policy_network: Any,
				env: Any, 
				discount: float, bp_discount: float,
				 unroll_length: int,
				 make_policy,
				 reward_function,
				 policy_batch_size=16):

	def step(carry: Tuple[Any, PRNGKey], tmp, policy):
		obs, env_state, key = carry # obs of shape  -1)
		key, transition_key, key_sample, key_reward = jax.random.split(key, 4)
		obs = detach(obs) + (bp_discount * obs) - (bp_discount * detach(obs))
		action, extra = policy(obs, key_sample) # action of shape ( -1)
		reward = reward_function(obs, action, key_reward)
		current_in = action
		next_state = env.step(env_state, action) # TODO: might need to vmap env.step
		next_obs = next_state.obs

		return (next_obs, next_state, key), (reward, obs, extra['entropy'])

	def get_imagined_out(policy_params, preprocess_params, init_obs, key):
		# init_obs of shape (-1)
		key, transition_key = jax.random.split(key)
		batch_size = policy_batch_size

		init_state = env.get_initial_state(init_obs, key)

		timesteps = jnp.arange(1, unroll_length + 1)
		f = functools.partial(step, 
							policy=make_policy((preprocess_params, policy_params)))
		(next_obs, _, _), (rewards, obs, entropy) = jax.lax.scan(f, (init_obs, init_state, key), timesteps)
		rewards = jnp.squeeze(rewards)
		trajectory_discounts = jnp.power(discount, jnp.arange(0, unroll_length))[jnp.newaxis, :]
		nstep_reward = jnp.sum(rewards * trajectory_discounts, axis=1) # (1,)

		imagined_return = nstep_reward
		
		total_entropy = jnp.mean(jnp.squeeze(entropy), axis=0) # entropies summed over episode. End shape (1,)

		return imagined_return, total_entropy

	batched_get_imagined_out = jax.vmap(get_imagined_out, in_axes=(None, None, 0, None), out_axes=(0, 0))

	def batch_policy_loss(policy_params, preprocess_params, init_obs, entropy_reg, key):
		# init obs of shape (policy_batch_size, 1, -1)
		imagined_return, total_entropies = batched_get_imagined_out(policy_params, preprocess_params,
															 jnp.reshape(init_obs, (policy_batch_size, -1)), key)
		averaged_returns = jnp.mean(imagined_return)
		total_entropy = jnp.mean(total_entropies)

		return -(averaged_returns + (entropy_reg*total_entropy)), {'entropy':total_entropy, 'img_ret': averaged_returns}


	return batch_policy_loss