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

from src.brax.exp3.offline_arm import losses, networks
from src.brax.exp3.offline_arm import true_policy_loss as tpl
from src.brax.exp3.offline_arm.evaluate_helper import *
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
	reward_params: Any
	transition_params: Any
	dynamics_optimizer_state: optax.OptState
	preprocessor_params: Any

def init_training_state(key, obs_size: int, arm_networks: networks.ARMNetworks,
						 dynamics_optimizer):
	key_policy, key_reward, key_transition, key_critic = jax.random.split(key, 4)
	
	reward_params = arm_networks.reward_network.init(key_reward)
	transition_params = arm_networks.transition_network.init(key_transition)
	dynamics_optimizer_state = dynamics_optimizer.init((transition_params, reward_params))
	
	training_state = TrainingState(
		reward_params=reward_params,
		transition_params=transition_params,
		dynamics_optimizer_state=dynamics_optimizer_state,
		preprocessor_params=None
	)
	return training_state


def get_dataset(env, length, smearing_rates, num_transitions, key):
	dataset = SeqReplayBuffer(num_transitions,
							 observation_dim=env.observation_size,
							 action_dim=env.action_size,
							 sampled_seq_len=length,
							 observation_type=np.float64)
	
	def get_short_term_optimal_action(env_state, smearing, key):
		smear_key, rkey, key = jax.random.split(key, 3)
		action = env.get_short_term_optimal_action(env_state, rkey)
		min_noise = -(action + 1)/smearing
		max_noise = (1 - action)/smearing
		action = action + (jax.random.truncated_normal(smear_key, lower=min_noise, upper=max_noise) * smearing)
		return jnp.squeeze(action)


	jit_reset = jax.jit(env.reset)
	jit_step = jax.jit(env.step)
	jit_get_short_term_optimal_action = jax.jit(get_short_term_optimal_action)

	ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []

	skey, key = jax.random.split(key)
	env_state = jit_reset(key)
	current_smearing = random.choice(smearing_rates)
	for i in range(num_transitions):
		ep_obs.append(env_state.obs)

		akey, key = jax.random.split(key)
		action = jit_get_short_term_optimal_action(env_state, smearing=current_smearing, key=akey)
		env_state = jit_step(env_state, action)

		ep_actions.append(action)
		ep_rewards.append(env_state.reward)
		ep_terminals.append(env_state.done)
		ep_next_obs.append(env_state.obs)

		if env_state.done:
			ep_actions = np.array(ep_actions)
			if len(ep_actions.shape) == 1:
				ep_actions = np.expand_dims(np.array(ep_actions), axis=-1)
			dataset.add_episode(np.array(ep_obs), np.array(ep_actions), 
										np.expand_dims(np.array(ep_rewards), axis=-1), 
										np.expand_dims(np.array(ep_terminals), axis=-1), 
										np.array(ep_next_obs))
			reset_key, key = jax.random.split(key)
			env_state = jit_reset(reset_key)
			ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []
			# reset policy chosen at the end of every episode for consistent policy throughout episode
			current_smearing = random.choice(smearing_rates)
	
	return dataset

def train(
		env: envs.Env,
		eval_env: envs.Env,
		episode_length: int,
		wm_steps: int,
		batch_size: int,
		action_repeat: int = 1,
		seed: int = 0,
		network_sizes=(64,64),
		epsilon=0.5,
		dynamics_lr=0.01,
		buffer_max=int(1e6),
		embd_dim: int = 48,
		progress_fn: Callable[[int, Any], None] = lambda *args: None,
		with_tqdm=False,
		input_observations=False,
		sequence_model_params={"name": "gpt",
							"transformer_nlayers" :3,
							"transformer_nheads": 3,
							"transformer_pdrop": 0.1}
		):

	print('experiment 3 version')

	key = jax.random.PRNGKey(seed)
	# env = wrappers.EpisodeWrapper(env, episode_length, action_repeat=action_repeat)
	eval_env = wrappers.EpisodeWrapper(eval_env, episode_length, action_repeat=action_repeat)

	obs_size = eval_env.observation_size
	action_size = eval_env.action_size

	max_episode_length = episode_length + 1 # Because we add initial observation to sequence of actions

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
	dynamics_loss = losses.make_losses(arm_networks,
																	input_observations=input_observations)

	dynamics_optimizer = optax.adam(learning_rate=dynamics_lr)

	# update functions
	dynamics_update = gradient_update_fn(dynamics_loss, dynamics_optimizer, has_aux=True, pmap_axis_name=None)


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

		metrics = {'tloss': daux['tloss']}

		return new_train_state, metrics


	# Jitting stuff
	eval_key, key = jax.random.split(key)
	evaluator = Evaluator(
						eval_env,
						partial(make_policy, deterministic=True),
						episode_length=episode_length,
						action_repeat=action_repeat,
						key=eval_key
						)

	init_training_state_key, eval_key, key = jax.random.split(key, 3)
	training_state = init_training_state(init_training_state_key, obs_size, arm_networks,
							 dynamics_optimizer)


	# jit_actor_step = jax.jit(actor_step)
	jit_train_wm = jax.jit(wm_training_step)
	wm_metrics = []
	total_policy_eval_steps = 0
	wm_total_steps = 0

	key, data_key = jax.random.split(key)
	smearing_rates = [0]
	replay_buffer = get_dataset(env, episode_length, smearing_rates, buffer_max, data_key)

	# TRAINING WORLD MODEL
	iterator = tqdm.tqdm(range(wm_steps)) if with_tqdm else range(wm_steps)
	reset_key, key = jax.random.split(key)

	for i in iterator:
		train_key, key = jax.random.split(key)
		sampled_episodes = replay_buffer.random_episodes(batch_size)
		training_state, wm_met = jit_train_wm(training_state, sampled_episodes, train_key)
		wm_metrics.append(wm_met)
		progress_fn(wm_total_steps, wm_met)
		wm_total_steps += 1


	return make_policy, (arm_networks, training_state),\
	 wm_metrics, replay_buffer


		

