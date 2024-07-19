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

from src.brax.exp3.true_pg import networks
from src.brax.exp3.true_pg import true_policy_loss as tpl
from src.brax.exp3.offline_arm.train import get_dataset, get_policy_dataset, add_policy_data
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
	preprocessor_params: Any

def init_training_state(key, obs_size: int, arm_networks: networks.ARMNetworks, policy_optimizer):
	key_policy, key_reward, key_transition, key_critic = jax.random.split(key, 4)
	
	policy_params = arm_networks.policy_network.init(key_policy)
	policy_optimizer_state = policy_optimizer.init(policy_params)

	
	training_state = TrainingState(
		policy_optimizer_state=policy_optimizer_state,
		policy_params=policy_params,
		preprocessor_params=None
	)
	return training_state



def train(
		env: envs.Env,
		eval_env: envs.Env,
		episode_length: int,
		unroll_length: int,
		num_cycles: int,
		data_per_cycle:int,
		max_policy_steps: int,
		policy_batch_size: int,
		eval_every: int = 10,
		action_repeat: int = 1,
		seed: int = 0,
		network_sizes=(64,64),
		bp_discount=0.99,
		discount=0.99,
		entropy_init=1000,
		entropy_decay_rate=0.99,
		entropy_transition_steps=500,
		policy_lr=0.001,
		grad_clip=10,
		buffer_max=int(1e6),
		embd_dim: int = 48,
		progress_fn: Callable[[int, Any], None] = lambda *args: None,
		with_tqdm=False,
		):

	print('experiment 2 version')

	key = jax.random.PRNGKey(seed)
	# env = wrappers.EpisodeWrapper(env, episode_length, action_repeat=action_repeat)
	eval_env = wrappers.EpisodeWrapper(eval_env, episode_length, action_repeat=action_repeat)

	obs_size = eval_env.observation_size
	action_size = eval_env.action_size

	max_episode_length = unroll_length + 1 # Because we add initial observation to sequence of actions

	# Make networks and define loss functions
	arm_networks = networks.make_arm_rnn_networks(obs_size,
											 action_size,
											 max_episode_length, 
											 hidden_layer_sizes=network_sizes,
											 decoder_hidden_layer_sizes=network_sizes,
											 embd_dim=embd_dim,
											 activation=flax.linen.swish,
											 input_observations=False,
											 seed=seed)

	make_policy = networks.make_inference_fn(arm_networks)
	true_policy_loss = tpl.make_loss(arm_networks.policy_network, env, discount, bp_discount,
													unroll_length=unroll_length//action_repeat, make_policy=make_policy,
													reward_function=env.make_reward_fn(batched=False), policy_batch_size=policy_batch_size,)

	policy_optimizer = optax.adam(learning_rate=policy_lr)
	# update functions
	true_policy_update = gradient_update_fn(true_policy_loss, policy_optimizer, has_aux=True, pmap_axis_name=None, max_gradient_norm=grad_clip)

	# entropy decay
	entropy_reg_fn = optax.exponential_decay(init_value=entropy_init, transition_steps=entropy_transition_steps, decay_rate=entropy_decay_rate)


	def true_policy_gradient(training_state: TrainingState, transitions: dict, entropy_reg: float, key):
		pkey, key = jax.random.split(key)

		sampled_init_obs = jnp.array(transitions['obs'])[:, 0:1, :]
		(ploss, paux), policy_params, policy_optimizer_state, p_grad = true_policy_update(training_state.policy_params,
																					training_state.preprocessor_params,
																					sampled_init_obs,
																					entropy_reg,
																					pkey,
																					optimizer_state=training_state.policy_optimizer_state)

		new_train_state = training_state.replace(
			policy_optimizer_state=policy_optimizer_state,
			policy_params=policy_params)

		metrics = {'img_ret': paux['img_ret'],
			 'entropy': paux['entropy'], 'grad_norms': optax.global_norm(p_grad),
			'ploss': ploss}

		return new_train_state, metrics

	def actor_step(env_state: envs.State, preprocessor_params: Any, policy_params: Any, key):
		"""Wrapper for acting.actor_step so it's jittable.
		Taking variable env from outside this scope
		"""
		policy = make_policy((preprocessor_params, policy_params))
		env_state, transition = acting.actor_step(env, env_state, policy, key)
		return env_state, transition

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
	training_state = init_training_state(init_training_state_key, obs_size, arm_networks, policy_optimizer)

	jit_reset = jax.jit(env.reset)
	jit_step = jax.jit(env.step)
	jit_actor_step = jax.jit(actor_step)
	jit_train_policy = jax.jit(true_policy_gradient)
	env_fns = (jit_reset, jit_step)
	policy_metrics = []
	total_policy_eval_steps = 0
	policy_steps = 0

	# GETTING DATA
	num_transitions = data_per_cycle * episode_length
	key, data_key = jax.random.split(key)
	replay_buffer = add_policy_data(env_fns, jit_actor_step, replay_buffer, episode_length, num_transitions, 
									make_policy, training_state.policy_params, training_state.preprocessor_params, data_key)

	for cycle in tqdm.tqdm(range(num_cycles)):
		# TRAINING POLICY
		incorrect_grads = 0
		num_policy_steps = 0
		while num_policy_steps < max_policy_steps:
			train_key, key = jax.random.split(key)
			sampled_episodes = replay_buffer.random_episodes(policy_batch_size)
			entropy_reg = entropy_reg_fn(policy_steps)
			training_state, policy_met = jit_train_policy(training_state, sampled_episodes, entropy_reg, train_key)

			if num_policy_steps % eval_every == 0:
				eval_metrics = evaluator.run_evaluation((training_state.preprocessor_params, training_state.policy_params), policy_met)
				policy_metrics.append(eval_metrics)
				progress_fn(total_policy_eval_steps, eval_metrics)
				total_policy_eval_steps += 1
			num_policy_steps += 1
			policy_steps += 1

	return make_policy, (training_state.preprocessor_params, training_state.policy_params), (arm_networks, training_state),\
	 policy_metrics, replay_buffer


		

