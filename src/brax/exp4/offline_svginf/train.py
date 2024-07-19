import tqdm
from typing import Any, Callable

import jax
import flax
import jax.numpy as jnp
from functools import partial
import gymnax
import optax
import random
import numpy as np
import functools

from src.brax.exp2.offline_svginf import losses, networks
from src.brax.exp2.offline_arm.train import get_dataset, get_policy_dataset, add_policy_data
from src.brax.exp2.offline_arm import true_policy_loss as tpl
from src.brax.exp2.offline_arm.networks import make_inference_fn
from src.brax.seq_replay_buffer import SeqReplayBuffer
from src.brax.gradients import gradient_update_fn
from src.misc.helper_methods import moving_avg
from src.brax.evaluator import Evaluator
from src.brax.custom_envs import wrappers
from src.misc.helper_methods import detach

from brax import envs
from brax.training import replay_buffers
from brax.training import acting

@flax.struct.dataclass
class TrainingState:
	policy_optimizer_state: optax.OptState
	policy_params: Any
	reward_optimizer_state: optax.OptState
	reward_params: Any
	transition_optimizer_state: optax.OptState
	transition_params: Any
	critic_params: Any
	critic_optimizer_state: optax.OptState
	preprocessor_params: Any
	

def init_training_state(key, obs_size: int, svg_networks: networks.SVGNetworks,
					   reward_optimizer, transition_optimizer, policy_optimizer, critic_optimizer):
	key_policy, key_reward, key_transition, key_critic = jax.random.split(key, 4)
	
	policy_params = svg_networks.policy_network.init(key_policy)
	policy_optimizer_state = policy_optimizer.init(policy_params)
	
	reward_params = svg_networks.reward_network.init(key_reward)
	reward_optimizer_state = reward_optimizer.init(reward_params)
	
	transition_params = svg_networks.transition_network.init(key_transition)
	transition_optimizer_state = transition_optimizer.init(transition_params)

	critic_params = svg_networks.critic_network.init(key_critic)
	critic_optimizer_state = critic_optimizer.init(critic_params)
	
	training_state = TrainingState(
		policy_optimizer_state=policy_optimizer_state,
		policy_params=policy_params,
		reward_optimizer_state=reward_optimizer_state,
		reward_params=reward_params,
		transition_optimizer_state=transition_optimizer_state,
		transition_params=transition_params,
		critic_params=critic_params,
		critic_optimizer_state=critic_optimizer_state,
		preprocessor_params=None
	)
	return training_state

class MultistepTransition:
    def __init__(self, transition_model):
        self.transition_model = transition_model
        
    def apply_sequence(self, preprocessor_params, transition_params, init_obs, actions, key, train=False):
        
        def step(carry, tmp, preprocess_params, transition_params):
            obs, key = carry
            key, key_sample, key_reward = jax.random.split(key, 3)
            action = tmp
            obs = jnp.squeeze(obs, axis=1)
            next_obs = self.transition_model.apply(preprocess_params, transition_params, obs, action)
            next_obs = jnp.expand_dims(next_obs, axis=1)
            return (next_obs, key), obs
        
        f = functools.partial(step, preprocess_params=preprocessor_params,
                              transition_params=transition_params,
                             )
        actions = jnp.transpose(actions, axes=(1, 0, 2))
        (next_obs, _), obs = jax.lax.scan(f, (init_obs, key), actions)
        obs = jnp.transpose(obs, axes=(1, 0, 2))
        all_obs = jnp.concatenate((obs, next_obs), axis=1)
        return all_obs

def train(env: envs.Env,
		eval_env: envs.Env,
		episode_length: int,
		unroll_length: int,
		num_cycles: int,
		data_per_cycle:int,
		wm_steps: int,
		policy_grad_tolerance: int,
		max_policy_steps: int,
		batch_size: int,
		reset_every=True,
		true_reward: bool = True,
		policy_batch_size: int = 1,
		eval_every: int = 0,
		action_repeat: int = 1,
		seed: int = 0,
		network_sizes=(64,64),
		bp_discount=0.99,
		discount=0.99,
		entropy_init=1000,
		entropy_decay_rate=0.99,
		entropy_transition_steps=500,
		dynamics_lr=0.01,
		policy_lr=0.001,
		critic_lr=0.001,
		grad_clip=10,
		buffer_max=int(1e6),
		bootstrap=1,
		progress_fn: Callable[[int, Any], None] = lambda *args: None,
		with_tqdm=False,
		):

	key = jax.random.PRNGKey(seed)
	# env = wrappers.EpisodeWrapper(env, episode_length, action_repeat=action_repeat)
	eval_env = wrappers.EpisodeWrapper(eval_env, episode_length, action_repeat=action_repeat)

	obs_size = env.observation_size
	action_size = env.action_size

	# Make networks and define loss functions
	svg_networks = networks.make_svg_networks(env.observation_size,
											 env.action_size, hidden_layer_sizes=network_sizes,
											 activation=flax.linen.swish)
	make_policy = make_inference_fn(svg_networks)
	transition_loss, reward_loss, policy_loss, critic_loss = losses.make_losses(svg_networks, discount=discount, bp_discount=bp_discount,
																   env=env, unroll_length=unroll_length, 
																   make_policy=make_policy,
																   reward_function=env.make_reward_fn() if true_reward else None,
																  policy_batch_size=policy_batch_size,
																  bootstrap=bootstrap)
	true_policy_loss = tpl.make_loss(svg_networks.policy_network, env, discount, bp_discount,
												unroll_length=unroll_length//action_repeat, make_policy=make_policy,
												reward_function=env.make_reward_fn(batched=False), policy_batch_size=policy_batch_size)
	# optimizers
	reward_optimizer = optax.adam(learning_rate=dynamics_lr)
	transition_optimizer = optax.adam(learning_rate=dynamics_lr)
	policy_optimizer = optax.adam(learning_rate=policy_lr)
	critic_optimizer = optax.adam(learning_rate=critic_lr)
	# update functions
	reward_update = gradient_update_fn(reward_loss, reward_optimizer, pmap_axis_name=None)
	transition_update = gradient_update_fn(transition_loss, transition_optimizer, pmap_axis_name=None)
	policy_update = gradient_update_fn(policy_loss, policy_optimizer, has_aux=True, pmap_axis_name=None, max_gradient_norm=grad_clip)
	critic_update = gradient_update_fn(critic_loss, critic_optimizer, has_aux=False, pmap_axis_name=None)
	true_policy_grad = jax.value_and_grad(true_policy_loss, has_aux=True)

	# entropy decay
	entropy_reg_fn = optax.exponential_decay(init_value=entropy_init, transition_steps=entropy_transition_steps, decay_rate=entropy_decay_rate)


	def wm_training_step(training_state: TrainingState, transitions: dict, key):
		# update environment models
		obs = jnp.array(transitions['obs'])
		next_obs = jnp.array(transitions['obs2'])
		actions = jnp.array(transitions['act'])
		rewards = jnp.array(transitions['rew'])

		if true_reward:
			rloss, reward_params, reward_optimizer_state = 0, training_state.reward_params, training_state.reward_optimizer_state
		else:
			rloss, reward_params, reward_optimizer_state, _ = reward_update(training_state.reward_params, 
																		  training_state.preprocessor_params,
																		obs, actions, rewards, 
																		optimizer_state=training_state.reward_optimizer_state)
		tloss, transition_params, transition_optimizer_state, _ = transition_update(training_state.transition_params,
																				training_state.preprocessor_params,
																				obs, actions, next_obs,
																				optimizer_state=training_state.transition_optimizer_state)

		new_train_state = training_state.replace(transition_params=transition_params,
												 reward_params=reward_params,
												 transition_optimizer_state=transition_optimizer_state,
												 reward_optimizer_state=reward_optimizer_state)

		metrics = {'rloss': rloss, 'tloss': tloss}

		return new_train_state, metrics

	def policy_training_step(training_state: TrainingState, transitions: dict, entropy_reg: float, key, other_policy_params=None):
		pkey, ckey = jax.random.split(key)
		sampled_init_obs = jnp.array(transitions['obs'][:, 0:1, :])
		(value, aux), policy_params, policy_optimizer_state, p_grad = policy_update(training_state.policy_params,
															 training_state.preprocessor_params,
															 training_state.transition_params,
															 training_state.reward_params,
															 training_state.critic_params,
															 sampled_init_obs,
															 entropy_reg,
															 pkey,
															 other_policy_params,
															optimizer_state=training_state.policy_optimizer_state)
		p_grad, _ = jax.flatten_util.ravel_pytree(p_grad)

		target_value = detach(aux['img_rew'])
		critic_loss, critic_params, critic_optimizer_state, _ = critic_update(training_state.critic_params,
																			training_state.preprocessor_params,
																			sampled_init_obs,
																			target_value,
																			ckey,
																			optimizer_state=training_state.critic_optimizer_state)

		new_train_state = training_state.replace(policy_optimizer_state=policy_optimizer_state,
												policy_params=policy_params,
												critic_params=critic_params,
												critic_optimizer_state=critic_optimizer_state)

		_, true_grad = true_policy_grad(training_state.policy_params,
										training_state.preprocessor_params,
										sampled_init_obs,
										entropy_reg,
										pkey)
		true_grad, _ = jax.flatten_util.ravel_pytree(true_grad)
		l2_grad_error = jnp.mean((p_grad - true_grad)**2)
		cosim_grad = jnp.dot(p_grad, true_grad)/(jnp.linalg.norm(p_grad) * jnp.linalg.norm(true_grad))

		metrics = {'img_ret': aux['img_ret'],
				   'entropy': aux['entropy'], 'grad_norms': optax.global_norm(p_grad),
				  'ploss': value, 'closs': critic_loss,
				'divergence': aux['divergence'],
				'l2_grad_error': l2_grad_error,
				'cosim_grad': cosim_grad}

		return new_train_state, metrics
	def actor_step(env_state: envs.State, preprocessor_params: Any, policy_params: Any, key):
		"""Wrapper for acting.actor_step so it's jittable.
		Taking variable env from outside this scope
		"""
		policy = make_policy((preprocessor_params, policy_params))
		env_state, transition = acting.actor_step(env, env_state, policy, key)
		return env_state, transition
    
	def reset_dynamics(training_state: TrainingState, key):
		key_reward, key_transition = jax.random.split(key)

		reward_params = svg_networks.reward_network.init(key_reward)
		transition_params = svg_networks.transition_network.init(key_transition)

		transition_optimizer_state = transition_optimizer.init(transition_params)
		reward_optimizer_state = reward_optimizer.init(reward_params)

		new_train_state = training_state.replace(
			transition_params=transition_params,
			transition_optimizer_state=transition_optimizer_state,
			reward_optimizer_state=reward_optimizer_state)
		return new_train_state


	# Jitting stuff
	replay_buffer = SeqReplayBuffer(buffer_max,
						 observation_dim=env.observation_size,
						 action_dim=env.action_size,
						 sampled_seq_len=episode_length,
						 observation_type=np.float64)
	eval_key, key = jax.random.split(key)
	evaluator = Evaluator(
						eval_env,
						partial(make_policy, deterministic=True),
						episode_length=episode_length,
						action_repeat=action_repeat,
						key=eval_key
						)

	init_training_state_key, eval_key, key = jax.random.split(key, 3)
	training_state = init_training_state(init_training_state_key, obs_size, svg_networks,
						   reward_optimizer, transition_optimizer, policy_optimizer, critic_optimizer)

	jit_reset = jax.jit(env.reset)
	jit_step = jax.jit(env.step)
	jit_actor_step = jax.jit(actor_step)
	jit_train_wm = jax.jit(wm_training_step)
	jit_train_policy = jax.jit(policy_training_step)
	env_fns = (jit_reset, jit_step)
	wm_metrics = []
	policy_metrics = []
	total_policy_eval_steps = 0
	wm_resets = []
	wm_total_steps = 0

	for cycle in range(num_cycles):
		# GETTING DATA
		print('GETTING DATA')
		num_transitions = data_per_cycle * episode_length
		key, data_key = jax.random.split(key)
		replay_buffer = add_policy_data(env_fns, jit_actor_step, replay_buffer, episode_length, num_transitions, 
									make_policy, training_state.policy_params, training_state.preprocessor_params, data_key)

		# TRAINING WORLD MODEL
		print('TRAINING WORLD MODEL')
		iterator = tqdm.tqdm(range(wm_steps)) if with_tqdm else range(wm_steps)
		reset_key, key = jax.random.split(key)
		if reset_every:
			training_state = reset_dynamics(training_state, reset_key)
		for i in iterator:
			train_key, key = jax.random.split(key)
			sampled_episodes = replay_buffer.random_episodes(batch_size)
			training_state, wm_met = jit_train_wm(training_state, sampled_episodes, train_key)
			wm_metrics.append(wm_met)
			progress_fn(wm_total_steps, wm_met)
			wm_total_steps += 1

		# TRAINING POLICY
		print('IMPROVING POLICY')
		incorrect_grads = 0
		num_policy_steps = 0
		while incorrect_grads < policy_grad_tolerance and num_policy_steps < max_policy_steps:
			train_key, key = jax.random.split(key)
			sampled_episodes = replay_buffer.random_episodes(policy_batch_size)
			entropy_reg = entropy_reg_fn(i)
			training_state, policy_met = jit_train_policy(training_state, sampled_episodes, entropy_reg, train_key)
			if policy_met['cosim_grad'] < 0.1:
				incorrect_grads += 1
			else:
				incorrect_grads = 0

			if num_policy_steps % eval_every == 0:
				eval_metrics = evaluator.run_evaluation((training_state.preprocessor_params, training_state.policy_params), policy_met)
				policy_metrics.append(eval_metrics)
				progress_fn(total_policy_eval_steps, eval_metrics)
				total_policy_eval_steps += 1
			num_policy_steps += 1
		wm_resets.append(total_policy_eval_steps)

	return make_policy, (training_state.preprocessor_params, training_state.policy_params), (svg_networks, training_state),\
	 wm_metrics, policy_metrics, replay_buffer, wm_resets




