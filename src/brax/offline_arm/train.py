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

from src.brax.offline_arm import losses, networks
from src.brax.offline_arm import true_policy_loss as tpl
from src.brax.offline_arm.evaluate_helper import *
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

def get_policy_dataset(env, length, num_transitions, make_policy, policy_params, preprocessor_params, key):
	dataset = SeqReplayBuffer(num_transitions,
							 observation_dim=env.observation_size,
							 action_dim=env.action_size,
							 sampled_seq_len=length,
							 observation_type=np.float64)
	def actor_step(env_state: envs.State, preprocessor_params: Any, policy_params: Any, key):
		"""Wrapper for acting.actor_step so it's jittable.
		Taking variable env from outside this scope
		"""
		policy = make_policy((preprocessor_params, policy_params))
		env_state, transition = acting.actor_step(env, env_state, policy, key)
		return env_state, transition
	jit_reset = jax.jit(env.reset)
	jit_step = jax.jit(env.step)
	jit_actor_step = jax.jit(actor_step)

	ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []

	skey, key = jax.random.split(key)
	env_state = jit_reset(key)
	for i in range(num_transitions):
		ep_obs.append(env_state.obs)

		akey, key = jax.random.split(key)
		env_state, transition = jit_actor_step(env_state, preprocessor_params,
									 policy_params, akey)
		action = transition.action

		ep_actions.append(action)
		ep_rewards.append(env_state.reward)
		ep_terminals.append(env_state.done)
		ep_next_obs.append(env_state.obs)

		if env_state.done:
			dataset.add_episode(np.array(ep_obs), np.array(ep_actions), 
										np.expand_dims(np.array(ep_rewards), axis=-1), 
										np.expand_dims(np.array(ep_terminals), axis=-1), 
										np.array(ep_next_obs))
			reset_key, key = jax.random.split(key)
			env_state = jit_reset(reset_key)
			ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []
			# reset policy chosen at the end of every episode for consistent policy throughout episode
	
	return dataset

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
            dataset.add_episode(np.array(ep_obs), np.expand_dims(np.array(ep_actions), axis=-1), 
                                        np.expand_dims(np.array(ep_rewards), axis=-1), 
                                        np.expand_dims(np.array(ep_terminals), axis=-1), 
                                        np.array(ep_next_obs))
            reset_key, key = jax.random.split(key)
            env_state = jit_reset(reset_key)
            ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []
            # reset policy chosen at the end of every episode for consistent policy throughout episode
            current_smearing = random.choice(smearing_rates)
    
    return dataset

def get_wm_metrics(train_data, test_data, off_data, networks_state, transition_network, sample_size, key):
	train_errors = get_pred_errors(networks_state, transition_network, train_data, sample_size, key)
	test_errors = get_pred_errors(networks_state, transition_network, test_data, sample_size, key)
	off_errors = get_pred_errors(networks_state, transition_network, off_data, sample_size, key)

	train_Js = get_batch_grads(networks_state, transition_network, train_data, sample_size, key)
	train_cosims, train_l2s, train_deviations = get_first_actiongrad_stats(train_Js)

	test_Js = get_batch_grads(networks_state, transition_network, test_data, sample_size, key)
	test_cosims, test_l2s, test_deviations = get_first_actiongrad_stats(test_Js)

	off_Js = get_batch_grads(networks_state, transition_network, off_data, sample_size, key)
	off_cosims, off_l2s, off_deviations = get_first_actiongrad_stats(off_Js)


	return (train_errors, test_errors, off_errors), (train_l2s, test_l2s, off_l2s), (train_deviations, test_deviations, off_deviations), (train_cosims, test_cosims, off_cosims)

def train(
		env: envs.Env,
		eval_env: envs.Env,
		episode_length: int,
		unroll_length: int,
		num_steps: int,
		policy_steps: int,
		batch_size: int,
		policy_batch_size: int,
		eval_every: int = 10,
		action_repeat: int = 1,
		seed: int = 0,
		network_sizes=(64,64),
		epsilon=0.5,
		bp_discount=0.99,
		discount=0.99,
		bootstrap=1,
		entropy_init=1000,
		entropy_decay_rate=0.99,
		entropy_transition_steps=500,
		dynamics_lr=0.01,
		policy_lr=0.001,
		critic_lr=0.001,
		tau=0.005,
		grad_clip=10,
		buffer_max=int(1e6),
		embd_dim: int = 48,
		progress_fn: Callable[[int, Any], None] = lambda *args: None,
		with_tqdm=False,
		reset_every=0,
		input_observations=False,
		sequence_model_params={"name": "gpt",
							"transformer_nlayers" :3,
							"transformer_nheads": 3,
							"transformer_pdrop": 0.1}
		):

	key = jax.random.PRNGKey(seed)
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
	dynamics_loss, policy_loss, critic_loss = losses.make_losses(arm_networks, discount=discount, bp_discount=bp_discount,
																	bootstrap=bootstrap,
																	unroll_length=unroll_length//action_repeat, 
																	 make_policy=make_policy,
																	reward_function=env.make_reward_fn(),
																	policy_batch_size=policy_batch_size,
																	epsilon=epsilon,
																	input_observations=input_observations)
	true_policy_loss = tpl.make_loss(arm_networks.policy_network, env, discount, bp_discount,
													unroll_length=unroll_length//action_repeat, make_policy=make_policy,
													reward_function=env.make_reward_fn(batched=False), policy_batch_size=policy_batch_size,)

	dynamics_optimizer = optax.adam(learning_rate=dynamics_lr)
	policy_optimizer = optax.adam(learning_rate=policy_lr)
	critic_optimizer = optax.adam(learning_rate=critic_lr)
	# update functions
	dynamics_update = gradient_update_fn(dynamics_loss, dynamics_optimizer, has_aux=True, pmap_axis_name=None)
	policy_update = gradient_update_fn(policy_loss, policy_optimizer, has_aux=True, pmap_axis_name=None, max_gradient_norm=grad_clip)
	# true_policy_update = gradient_update_fn(true_policy_loss, policy_optimizer, has_aux=True, pmap_axis_name=None)
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
		p_grad, _= jax.flatten_util.ravel_pytree(p_grad)
		# update critic
		target_value = detach(paux['target_value'])
		critic_loss, critic_params, critic_optimizer_state, _ = critic_update(training_state.critic_params,
																		training_state.preprocessor_params,
																		sampled_init_obs,
																		target_value,
																		ckey,
																		optimizer_state=training_state.critic_optimizer_state)
		# update target critic
		target_critic_params = target_update(critic_params, training_state.target_critic_params, tau)

		# true policy grad
		value, true_grad = true_policy_grad(training_state.policy_params,
										training_state.preprocessor_params,
										sampled_init_obs,
										entropy_reg,
										pkey)
		true_grad, _ = jax.flatten_util.ravel_pytree(true_grad)
		l2_grad_error = jnp.mean((p_grad - true_grad)**2)
		cosim_grad = jnp.dot(p_grad, true_grad)/(jnp.linalg.norm(p_grad) * jnp.linalg.norm(true_grad))

		new_train_state = training_state.replace(
			policy_optimizer_state=policy_optimizer_state,
			policy_params=policy_params,
			critic_params=critic_params,
			target_critic_params=target_critic_params,
			critic_optimizer_state=critic_optimizer_state)

		metrics = {'img_ret': paux['img_ret'],
			 'entropy': paux['entropy'], 'grad_norms': optax.global_norm(p_grad),
			'ploss': ploss, 'closs': critic_loss, 
			'divergence': paux['divergence'],
			'l2_grad_error': l2_grad_error,
			'cosim_grad': cosim_grad}

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


	def reset_dynamics(training_state: TrainingState, key):
		key_reward, key_transition = jax.random.split(key)

		# reward_params = arm_networks.reward_network.init(key_reward)
		transition_params = arm_networks.transition_network.init(key_transition)

		dynamics_optimizer_state = dynamics_optimizer.init((transition_params, reward_params))

		new_train_state = training_state.replace(
			transition_params=transition_params,
			# reward_params=reward_params,
			dynamics_optimizer_state=dynamics_optimizer_state)

		return new_train_state

	# Jitting stuff
	jit_train_wm = jax.jit(wm_training_step)
	jit_train_policy = jax.jit(policy_training_step)
	# jit_train_policy = jax.jit(true_policy_gradient)


	init_training_state_key, eval_key, key = jax.random.split(key, 3)
	training_state = init_training_state(init_training_state_key, obs_size, arm_networks,
							 dynamics_optimizer, policy_optimizer, critic_optimizer)

	# For evaluation
	evaluator = Evaluator(
							eval_env,
							partial(make_policy, deterministic=True),
							episode_length=episode_length,
							action_repeat=action_repeat,
							key=eval_key
							)
	# Filling offline data
	print('Filling dataset with random* trajectories')
	key, data_key = jax.random.split(key)
	smearing_rates = [0., 0.5]
	replay_buffer = get_dataset(env, episode_length, smearing_rates, buffer_max, data_key)
	# replay_buffer = get_policy_dataset(env, episode_length, buffer_max, make_policy, 
									# training_state.policy_params, training_state.preprocessor_params, data_key)
	initial_policy_params = training_state.policy_params.copy({})

	wm_metrics = []
	print('Training world model')
	iterator = tqdm.tqdm(range(num_steps)) if with_tqdm else range(num_steps)
	for i in iterator:
		train_key, key = jax.random.split(key)
		sampled_episodes = replay_buffer.random_episodes(batch_size)
		training_state, wm_met = jit_train_wm(training_state, sampled_episodes, train_key)
		wm_metrics.append(wm_met)
		progress_fn(i, wm_met)

	policy_metrics = []
	print('Training policy')
	p_itr = tqdm.tqdm(range(policy_steps)) if with_tqdm else range(policy_steps)
	for i in p_itr:
		train_key, key = jax.random.split(key)
		sampled_episodes = replay_buffer.random_episodes(policy_batch_size)
		entropy_reg = entropy_reg_fn(i)
		training_state, policy_met = jit_train_policy(training_state, sampled_episodes, entropy_reg, train_key, other_policy_params=initial_policy_params)

		if i % eval_every == 0:
			eval_metrics = evaluator.run_evaluation((training_state.preprocessor_params, training_state.policy_params), policy_met)
			policy_metrics.append(eval_metrics)
			progress_fn(i, eval_metrics)

	return make_policy, (training_state.preprocessor_params, training_state.policy_params), (arm_networks.transition_network, training_state), wm_metrics, policy_metrics, replay_buffer


		


