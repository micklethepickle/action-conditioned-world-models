import tqdm
from typing import Any, Callable

import jax
import flax
import jax.numpy as jnp
from functools import partial
import gymnax
import optax
import numpy as np
import random

from src.brax.arm import losses, networks
from src.brax.arm import true_policy_loss as tpl
from src.brax.seq_replay_buffer import SeqReplayBuffer
from src.brax.gradients import gradient_update_fn
from src.misc.helper_methods import moving_avg
from src.brax.evaluator import Evaluator
from src.brax.custom_envs import wrappers
from src.misc.helper_methods import detach, target_update

from brax import envs
from brax.training import replay_buffers
from brax.training import acting

from src.brax.offline_arm.evaluate_helper import *

@flax.struct.dataclass
class TrainingState:
	policy_optimizer_state: optax.OptState
	policy_params: Any
	reward_params: Any
	transition_params: Any
	dynamics_optimizer_state: optax.OptState
	critic_params: Any
	target_critic_params: Any
	critic_optimizer_state: optax.OptState
	preprocessor_params: Any

def init_training_state(key, obs_size: int, arm_networks: networks.ARMNetworks,
						 dynamics_optimizer, policy_optimizer, critic_optimizer):
	key_policy, key_reward, key_transition, key_critic = jax.random.split(key, 4)
	
	policy_params = arm_networks.policy_network.init(key_policy)
	policy_optimizer_state = policy_optimizer.init(policy_params)
	
	reward_params = arm_networks.reward_network.init(key_reward)
	transition_params = arm_networks.transition_network.init(key_transition)
	dynamics_optimizer_state = dynamics_optimizer.init((transition_params, reward_params))

	critic_params = arm_networks.critic_network.init(key_critic)
	target_critic_params = detach(critic_params)
	critic_optimizer_state = critic_optimizer.init(critic_params)
	
	training_state = TrainingState(
		policy_optimizer_state=policy_optimizer_state,
		policy_params=policy_params,
		reward_params=reward_params,
		transition_params=transition_params,
		dynamics_optimizer_state=dynamics_optimizer_state,
		critic_params=critic_params,
		target_critic_params=target_critic_params,
		critic_optimizer_state=critic_optimizer_state,
		preprocessor_params=None
	)
	return training_state

def train(
		env: envs.Env,
		eval_env: envs.Env,
		episode_length: int,
		unroll_length: int,
		num_steps: int,
		batch_size: int,
		policy_batch_size: int,
		true_reward: bool = True,
		true_timesteps: bool = True,
		action_repeat: int = 1,
		seed: int = 0,
		network_sizes=(64,64),
		epsilon=0.9,
		bp_discount=0.99,
		discount=0.99,
		bootstrap=1,
		entropy_init=0.01,
		entropy_decay_rate=1,
		entropy_transition_steps=500,
		dynamics_lr=0.001,
		policy_lr=0.001,
		critic_lr=0.001,
		tau=0.005,
		grad_clip=1000,
		buffer_max=int(1e6),
		embd_dim: int = 48,
		progress_fn: Callable[[int, Any], None] = lambda *args: None,
		with_tqdm=False,
		reset_every=0,
		input_observations=False,
		true_pg=False,
		sequence_model_params={"name": "gpt",
							"transformer_nlayers" :3,
							"transformer_nheads": 3,
							"transformer_pdrop": 0.1}
		):

	print('experiment 2 version')

	key = jax.random.PRNGKey(seed)
	# env = wrappers.EpisodeWrapper(env, episode_length, action_repeat=action_repeat)
	eval_env = wrappers.EpisodeWrapper(eval_env, episode_length, action_repeat=action_repeat)

	obs_size = eval_env.observation_size
	action_size = eval_env.action_size

	max_episode_length = unroll_length + 1 # Because we add initial observation to sequence of actions

	# Make networks and define loss functions
	try:
		sequence_model_name = sequence_model_params.pop('name')
	except AttributeError:
		sequence_model_params = sequence_model_params.to_dict()
		sequence_model_name = sequence_model_params.pop('name')
	if sequence_model_name == 'gpt':
		arm_networks = networks.make_arm_networks(obs_size,
												 action_size,
												 max_episode_length, 
												 true_timesteps=true_timesteps,
												 hidden_layer_sizes=network_sizes,
												 decoder_hidden_layer_sizes=network_sizes,
												 embd_dim=embd_dim,
												 activation=flax.linen.swish,
												 seed=seed,
												 input_observations=input_observations,
												 **sequence_model_params
											 )
	elif sequence_model_name == 'lstm':
		arm_networks = networks.make_arm_lstm_networks(obs_size,
												 action_size,
												 max_episode_length, 
												 true_timesteps=true_timesteps,
												 hidden_layer_sizes=network_sizes,
												 decoder_hidden_layer_sizes=network_sizes,
												 embd_dim=embd_dim,
												 activation=flax.linen.swish,
												 input_observations=input_observations,
												 seed=seed)
	elif sequence_model_name == 'rnn':
		arm_networks = networks.make_arm_rnn_networks(obs_size,
												 action_size,
												 max_episode_length, 
												 true_timesteps=true_timesteps,
												 hidden_layer_sizes=network_sizes,
												 decoder_hidden_layer_sizes=network_sizes,
												 embd_dim=embd_dim,
												 activation=flax.linen.swish,
												 input_observations=input_observations,
												 seed=seed)
	elif sequence_model_name == 's4':
		arm_networks = networks.make_arm_s4_networks(obs_size,
													action_size,
													max_episode_length,
													true_timesteps=true_timesteps,
													hidden_layer_sizes=network_sizes,
													decoder_hidden_layer_sizes=network_sizes,
													embd_dim=embd_dim,
													activation=flax.linen.swish,
													input_observations=input_observations,
													seed=seed,
													**sequence_model_params)
	else:
		raise ValueError('Incorrect sequence model name')

	make_policy = networks.make_inference_fn(arm_networks)
	dynamics_loss, policy_loss, critic_loss = losses.make_losses(arm_networks, discount=discount, bp_discount=bp_discount,
																	bootstrap=bootstrap,
																	unroll_length=unroll_length//action_repeat, 
																	 make_policy=make_policy,
																	reward_function=env.make_reward_fn() if true_reward else None,
																	policy_batch_size=policy_batch_size,
																	epsilon=epsilon,
																	input_observations=input_observations,
																	deterministic_policy=True)
	true_policy_loss = tpl.make_loss(arm_networks.policy_network, env, discount, bp_discount,
													unroll_length=unroll_length//action_repeat, make_policy=make_policy,
													reward_function=env.make_reward_fn(batched=False), policy_batch_size=policy_batch_size,)
	deterministic_tpl = tpl.make_loss(arm_networks.policy_network, env, discount, bp_discount,
													unroll_length=unroll_length//action_repeat, make_policy=make_policy,
													reward_function=env.make_reward_fn(batched=False), policy_batch_size=policy_batch_size,
													deterministic_policy=True)

	dynamics_optimizer = optax.adam(learning_rate=dynamics_lr)
	policy_optimizer = optax.adam(learning_rate=policy_lr)
	dummy_optimizer = optax.adam(learning_rate=0.)
	critic_optimizer = optax.adam(learning_rate=critic_lr)
	# update functions
	dynamics_update = gradient_update_fn(dynamics_loss, dynamics_optimizer, has_aux=True, pmap_axis_name=None)
	policy_update = gradient_update_fn(policy_loss, dummy_optimizer, has_aux=True, pmap_axis_name=None, max_gradient_norm=grad_clip)
	true_policy_update = gradient_update_fn(true_policy_loss, policy_optimizer, has_aux=True, pmap_axis_name=None)
	deterministic_tpg = jax.value_and_grad(deterministic_tpl, has_aux=True)
	critic_update = gradient_update_fn(critic_loss, critic_optimizer, has_aux=False, pmap_axis_name=None)


	# entropy decay
	entropy_reg_fn = optax.exponential_decay(init_value=entropy_init, transition_steps=entropy_transition_steps, decay_rate=entropy_decay_rate)

	def wm_training_step(training_state: TrainingState, transitions: dict, key):

		dkey, key = jax.random.split(key)
		obs = jnp.array(transitions['obs'])
		next_obs = jnp.array(transitions['obs2'])
		actions = jnp.array(transitions['act'])
		rewards = jnp.array(transitions['rew'])

		(dloss, daux), (transition_params, reward_params), dynamics_optimizer_state, _ = dynamics_update((training_state.transition_params, training_state.reward_params),
																										training_state.preprocessor_params,
																										obs, actions, rewards, next_obs, dkey,
																										optimizer_state=training_state.dynamics_optimizer_state)

		new_train_state = training_state.replace(transition_params=transition_params,
												reward_params=reward_params,
												dynamics_optimizer_state=dynamics_optimizer_state)

		metrics = {'rloss': daux['rloss'], 'tloss': daux['tloss']}

		return new_train_state, metrics

	def policy_training_step(training_state: TrainingState, transitions:dict, entropy_reg: float, key, other_policy_params=None):
		pkey, ckey, key = jax.random.split(key, 3)

		sampled_init_obs = jnp.array(transitions['obs'])[:, 0:1, :]
		(ploss, paux), policy_params, policy_optimizer_state, p_grad = policy_update(training_state.policy_params,
																	 training_state.preprocessor_params,
																	 training_state.transition_params,
																	 training_state.reward_params,
																	 training_state.critic_params,
																	 training_state.target_critic_params,
																	 sampled_init_obs,
																	 entropy_reg,
																	 pkey,
																	 other_policy_params,
																	optimizer_state=training_state.policy_optimizer_state)
		p_grad, _ = jax.flatten_util.ravel_pytree(p_grad)

		# true policy grad
		value, true_grad = deterministic_tpg(training_state.policy_params,
										training_state.preprocessor_params,
										sampled_init_obs,
										entropy_reg,
										pkey)
		true_grad, _ = jax.flatten_util.ravel_pytree(true_grad)
		l2_grad_error = jnp.mean((p_grad - true_grad)**2)
		cosim_grad = jnp.dot(p_grad, true_grad)/(jnp.linalg.norm(p_grad) * jnp.linalg.norm(true_grad))

		new_train_state = training_state.replace(
			policy_optimizer_state=policy_optimizer_state,
			policy_params=policy_params)

		metrics = {'img_ret': paux['img_ret'],
			 'entropy': paux['entropy'], 'grad_norms': optax.global_norm(p_grad),
			'ploss': ploss,
			'divergence': paux['divergence'],
			'l2_grad_error': l2_grad_error,
			'cosim_grad': cosim_grad,
			'tpg_value': value}

		return new_train_state, metrics

	def true_policy_gradient(training_state: TrainingState, transitions: dict, entropy_reg: float, key):
		pkey, key = jax.random.split(key)

		sampled_init_obs = jnp.array(transitions['obs'])[:, 0:1, :]
		(ploss, paux), policy_params, policy_optimizer_state, p_grad_norms = true_policy_update(training_state.policy_params,
																					training_state.preprocessor_params,
																					sampled_init_obs,
																					entropy_reg,
																					pkey,
																					optimizer_state=training_state.policy_optimizer_state)

		new_train_state = training_state.replace(
			policy_optimizer_state=policy_optimizer_state,
			policy_params=policy_params)

		metrics = {'img_ret': paux['img_ret'],
			 'entropy': paux['entropy'], 'grad_norms': p_grad_norms,
			'ploss': ploss}

		return new_train_state, metrics

	def make_exploration_policy(preprocessor_params, policy_params):
		policy = make_policy((preprocessor_params, policy_params), deterministic=True)
		def expl_policy(obs, key):
			actions, extra = policy(obs, key)
			noise_key, key = jax.random.split(key)
			actions = actions + (jax.random.normal(noise_key, actions.shape) * 0.1)
			return actions, extra
		return expl_policy

	def actor_step(env_state: envs.State, preprocessor_params: Any, policy_params: Any, key):
		"""Wrapper for acting.actor_step so it's jittable.
		Taking variable env from outside this scope
		"""
		policy = make_exploration_policy(preprocessor_params, policy_params)
		env_state, transition = acting.actor_step(env, env_state, policy, key)
		return env_state, transition

	def reset_dynamics(training_state: TrainingState, key):
		key_reward, key_transition = jax.random.split(key)

		reward_params = arm_networks.reward_network.init(key_reward)
		transition_params = arm_networks.transition_network.init(key_transition)

		new_train_state = training_state.replace(
			reward_params=reward_params,
			transition_params=transition_params)

		return new_train_state

	# Jitting stuff
	eval_key, key = jax.random.split(key)
	evaluator = Evaluator(
						eval_env,
						partial(make_policy, deterministic=True),
						episode_length=episode_length,
						action_repeat=action_repeat,
						key=eval_key
						)

	init_training_state_key, env_reset_key, key = jax.random.split(key, 3)
	training_state = init_training_state(init_training_state_key, obs_size, arm_networks,
							 dynamics_optimizer, policy_optimizer, critic_optimizer)

	jit_reset = jax.jit(env.reset)
	jit_actor_step = jax.jit(actor_step)
	jit_train_wm = jax.jit(wm_training_step)
	jit_reset_dynamics = jax.jit(reset_dynamics)

	# reset replay buffer and gather data just from this slightly trained policy
	replay_buffer = SeqReplayBuffer(buffer_max,
					 observation_dim=env.observation_size,
					 action_dim=env.action_size,
					 sampled_seq_len=episode_length,
					 observation_type=np.float64)
	env_state = jit_reset(env_reset_key)
	ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []

	for i in range(buffer_max - 1):
		exp_key, key = jax.random.split(key)
		env_state, transition = jit_actor_step(env_state, training_state.preprocessor_params,
											 training_state.policy_params, exp_key)
		ep_obs.append(transition.observation)
		ep_actions.append(transition.action)
		ep_rewards.append(transition.reward)
		ep_terminals.append(env_state.done)
		ep_next_obs.append(transition.next_observation)

		if env_state.done:
			replay_buffer.add_episode(np.array(ep_obs), np.array(ep_actions), 
										np.expand_dims(np.array(ep_rewards), axis=-1), 
										np.expand_dims(np.array(ep_terminals), axis=-1), 
										np.array(ep_next_obs))
			reset_key, key = jax.random.split(key)
			env_state = jit_reset(reset_key)
			ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []

	# train world model on filled replay buffer
	all_wm_metrics = []
	iterator = tqdm.tqdm(range(num_steps)) if with_tqdm else range(num_steps)
	for i in iterator:
		train_key, key = jax.random.split(key)
		sampled_episodes = replay_buffer.random_episodes(batch_size)
		training_state, wm_metrics = jit_train_wm(training_state, sampled_episodes, train_key)
		progress_fn(i, wm_metrics)
		all_wm_metrics.append(wm_metrics)

	# check policy grad accuracy
	jit_train_policy = jax.jit(policy_training_step)
	grad_metrics = []

	iterator = tqdm.tqdm(range(5)) if with_tqdm else range(5)
	for i in iterator:
		train_key, key = jax.random.split(key)
		policy_sampled_episodes = replay_buffer.random_episodes(policy_batch_size)
		entropy_reg = entropy_reg_fn(i)
		training_state, policy_metrics = jit_train_policy(training_state, policy_sampled_episodes, entropy_reg, train_key)

		metrics = policy_metrics
		eval_metrics = evaluator.run_evaluation((training_state.preprocessor_params, training_state.policy_params), metrics)
		grad_metrics.append(eval_metrics)

	# test prediction error
	test_key, key = jax.random.split(key)
	test_dataset = SeqReplayBuffer(buffer_max,
					 observation_dim=env.observation_size,
					 action_dim=env.action_size,
					 sampled_seq_len=episode_length,
					 observation_type=np.float64)
	env_state = jit_reset(env_reset_key)
	ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []

	for i in range(100*episode_length):
		exp_key, key = jax.random.split(key)
		env_state, transition = jit_actor_step(env_state, training_state.preprocessor_params,
											 training_state.policy_params, exp_key)
		ep_obs.append(transition.observation)
		ep_actions.append(transition.action)
		ep_rewards.append(transition.reward)
		ep_terminals.append(env_state.done)
		ep_next_obs.append(transition.next_observation)

		if env_state.done:
			test_dataset.add_episode(np.array(ep_obs), np.array(ep_actions), 
										np.expand_dims(np.array(ep_rewards), axis=-1), 
										np.expand_dims(np.array(ep_terminals), axis=-1), 
										np.array(ep_next_obs))
			reset_key, key = jax.random.split(key)
			env_state = jit_reset(reset_key)
			ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []
	test_errors = get_pred_errors(training_state, arm_networks.transition_network, test_dataset, 100, test_key)

	return make_policy, (arm_networks, training_state), (grad_metrics, all_wm_metrics, test_errors), replay_buffer


		

