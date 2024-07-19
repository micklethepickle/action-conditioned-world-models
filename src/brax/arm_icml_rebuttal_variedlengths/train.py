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
import math

from src.brax.arm_icml_rebuttal_variedlengths import losses, networks
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
		warmup_steps: int,
		dynamics_update_every: int,
		policy_update_every: int,
		batch_size: int,
		policy_batch_size: int,
		true_reward: bool = True,
		true_timesteps: bool = True,
		eval_every: int = 10,
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
		detach_obs=False,
		difference_transition=False,
		stochastic_transition=False,
		eval_batch_size=32,
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
	dynamics_loss, _, critic_loss = losses.make_losses(arm_networks, discount=discount, bp_discount=bp_discount,
																	bootstrap=bootstrap,
																	 make_policy=make_policy,
																	reward_function=env.make_reward_fn() if true_reward else None,
																	policy_batch_size=policy_batch_size,
																	epsilon=epsilon,
																	input_observations=input_observations,
																	detach_obs=detach_obs,
																	unroll_length=unroll_length,
																	difference_transition=difference_transition,
																	stochastic_transition=stochastic_transition)
	policy_losses = []
	for i in range(episode_length):
		# Make a different policy loss for each unroll length possibility
		_, plos, _ = losses.make_losses(arm_networks, discount=discount, bp_discount=bp_discount,
																	bootstrap=bootstrap,
																	 make_policy=make_policy,
																	reward_function=env.make_reward_fn() if true_reward else None,
																	policy_batch_size=policy_batch_size,
																	epsilon=epsilon,
																	input_observations=input_observations,
																	detach_obs=detach_obs,
																	unroll_length=unroll_length - i,
																	difference_transition=difference_transition,
																	stochastic_transition=stochastic_transition)
		policy_losses.append(plos)
	
	true_policy_loss = tpl.make_loss(arm_networks.policy_network, env, discount, bp_discount,
													unroll_length=unroll_length//action_repeat, make_policy=make_policy,
													reward_function=env.make_reward_fn(batched=False), policy_batch_size=policy_batch_size,)

	dynamics_optimizer = optax.adam(learning_rate=dynamics_lr)
	policy_optimizer = optax.adam(learning_rate=policy_lr)
	critic_optimizer = optax.adam(learning_rate=critic_lr)
	# update functions
	dynamics_update = gradient_update_fn(dynamics_loss, dynamics_optimizer, has_aux=True, pmap_axis_name=None)
	all_policy_updates = [jax.jit(gradient_update_fn(pl, policy_optimizer, has_aux=True, pmap_axis_name=None, max_gradient_norm=grad_clip)) for pl in policy_losses]
	# policy_update = gradient_update_fn(policy_loss, policy_optimizer, has_aux=True, pmap_axis_name=None, max_gradient_norm=grad_clip)
	true_policy_update = gradient_update_fn(true_policy_loss, policy_optimizer, has_aux=True, pmap_axis_name=None)
	true_policy_grad = jax.value_and_grad(true_policy_loss, has_aux=True)
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

	def policy_training_step(training_state: TrainingState, transitions:dict, entropy_reg: float, key, start_index=0, other_policy_params=None):
		pkey, ckey, key = jax.random.split(key, 3)

		sampled_init_obs = jnp.array(transitions['obs'])[:, 0:1, :]
		(ploss, paux), policy_params, policy_optimizer_state, p_grad = all_policy_updates[start_index](training_state.policy_params,
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
		p_grad, _= jax.flatten_util.ravel_pytree(p_grad)

		new_train_state = training_state.replace(
			policy_optimizer_state=policy_optimizer_state,
			policy_params=policy_params)

		metrics = {'img_ret': paux['img_ret'],
			 'entropy': paux['entropy'], 'grad_norms': optax.global_norm(p_grad),
			'ploss': ploss,
			'divergence': paux['divergence']}

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
		p_grad_norms = optax.global_norm(p_grad_norms)
		new_train_state = training_state.replace(
			policy_optimizer_state=policy_optimizer_state,
			policy_params=policy_params)

		metrics = {'img_ret': paux['img_ret'],
			 'entropy': paux['entropy'], 'grad_norms': p_grad_norms,
			'ploss': ploss}

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

		reward_params = arm_networks.reward_network.init(key_reward)
		transition_params = arm_networks.transition_network.init(key_transition)

		new_train_state = training_state.replace(
			reward_params=reward_params,
			transition_params=transition_params)

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

	init_training_state_key, env_reset_key, key = jax.random.split(key, 3)
	training_state = init_training_state(init_training_state_key, obs_size, arm_networks,
							 dynamics_optimizer, policy_optimizer, critic_optimizer)

	jit_reset = jax.jit(env.reset)
	jit_actor_step = jax.jit(actor_step)
	jit_train_wm = jax.jit(wm_training_step)
	# jit_train_policy = jax.jit(policy_training_step)
	jit_reset_dynamics = jax.jit(reset_dynamics)


	ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []
	all_returns = []
	all_metrics = []
	metrics = {}

	env_state = jit_reset(env_reset_key)
	iterator = tqdm.tqdm(range(num_steps)) if with_tqdm else range(num_steps)
	for i in iterator:
		# take step
		exp_key, key = jax.random.split(key)
		env_state, transition = jit_actor_step(env_state, training_state.preprocessor_params,
											 training_state.policy_params, exp_key)
		ep_obs.append(transition.observation)
		ep_actions.append(transition.action)
		ep_rewards.append(transition.reward)
		ep_terminals.append(env_state.done)
		ep_next_obs.append(transition.next_observation)
		
		# save episode in replay buffer and reset
		if env_state.done:
			replay_buffer.add_episode(np.array(ep_obs), np.array(ep_actions), 
										np.expand_dims(np.array(ep_rewards), axis=-1), 
										np.expand_dims(np.array(ep_terminals), axis=-1), 
										np.array(ep_next_obs))
			reset_key, key = jax.random.split(key)
			env_state = jit_reset(reset_key)
			all_returns.append(np.sum(ep_rewards))
			ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []

		# Reset transition network if necessary
		if reset_every > 0 and i % reset_every == 0 and i > warmup_steps:
			reset_dynamics_key, key = jax.random.split(key)
			training_state = jit_reset_dynamics(training_state, reset_dynamics_key)
		
		# update
		if i % dynamics_update_every == 0 and i > warmup_steps:	
			train_key, key = jax.random.split(key)
			all_wm_metrics = []
			for j in range(episode_length):
				sampled_episodes = replay_buffer.random_episodes(batch_size, start_index=j)
				train_key, key = jax.random.split(key)
				training_state, wm_metrics = jit_train_wm(training_state, sampled_episodes, train_key)
				all_wm_metrics.append(wm_metrics)
			wm_metrics = {k: jnp.mean(jnp.array([m[k] for m in all_wm_metrics])) for k in all_wm_metrics[0]}
			# sampled_episodes = replay_buffer.random_episodes(batch_size)
			# training_state, wm_metrics = jit_train_wm(training_state, sampled_episodes, train_key)

		if i % policy_update_every == 0 and i > warmup_steps:
			train_key, key = jax.random.split(key)
			all_policy_metrics = []
			for j in range(episode_length):
				policy_sampled_episodes = replay_buffer.random_episodes(policy_batch_size, start_index=j)
				train_key, key = jax.random.split(key)
				entropy_reg = entropy_reg_fn(i)
				training_state, policy_metrics = policy_training_step(training_state, policy_sampled_episodes, entropy_reg, train_key, start_index=j)
				all_policy_metrics.append(policy_metrics)

			policy_metrics = {k: jnp.mean(jnp.array([m[k] for m in all_policy_metrics])) for k in all_policy_metrics[0]}
			# entropy_reg = entropy_reg_fn(i)
			# training_state, policy_metrics = policy_training_step(training_state, policy_sampled_episodes, entropy_reg, train_key, start_index=0)

		# Run evals
		# if i % eval_every == 0 and i > warmup_steps:
		# 	metrics = policy_metrics
		# 	if not true_pg:
		# 		for k in wm_metrics:
		# 			metrics[k] = wm_metrics[k]
		# 	metrics['train/episode_reward'] = all_returns[-1]
		# 	eval_metrics = evaluator.run_evaluation((training_state.preprocessor_params, training_state.policy_params), metrics)
		# 	all_metrics.append(eval_metrics)
		# 	progress_fn(i, eval_metrics)

		if i % eval_every == 0 and i > warmup_steps:
			metrics = policy_metrics
			for k in wm_metrics:
				metrics[k] = wm_metrics[k]
			metrics['train/episode_reward'] = all_returns[-1]
			all_eval_metrics = []
			for j in range(eval_batch_size):
				eval_metrics = evaluator.run_evaluation((training_state.preprocessor_params, training_state.policy_params), metrics)
				all_eval_metrics.append(eval_metrics)
			eval_metrics = {k: jnp.mean(jnp.array([m[k] for m in all_eval_metrics])) for k in all_eval_metrics[0]}
			all_metrics.append(eval_metrics)
			progress_fn(i, eval_metrics)


	return make_policy, (arm_networks, training_state), all_metrics, replay_buffer


		

