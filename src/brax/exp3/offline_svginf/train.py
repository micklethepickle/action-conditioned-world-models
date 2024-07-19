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

from src.brax.exp3.offline_svginf import losses, networks
from src.brax.exp3.offline_arm.train import get_dataset
from src.brax.exp3.offline_arm import true_policy_loss as tpl
from src.brax.exp3.offline_arm.networks import make_inference_fn
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
	transition_optimizer_state: optax.OptState
	transition_params: Any
	preprocessor_params: Any
	

def init_training_state(key, obs_size: int, svg_networks: networks.SVGNetworks, transition_optimizer):
	key_policy, key_reward, key_transition, key_critic = jax.random.split(key, 4)
	
	
	
	transition_params = svg_networks.transition_network.init(key_transition)
	transition_optimizer_state = transition_optimizer.init(transition_params)

	
	training_state = TrainingState(
		transition_optimizer_state=transition_optimizer_state,
		transition_params=transition_params,
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
		wm_steps: int,
		batch_size: int,
		action_repeat: int = 1,
		seed: int = 0,
		network_sizes=(64,64),
		dynamics_lr=0.01,
		buffer_max=int(1e6),
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
	transition_loss = losses.make_losses(svg_networks,
											   env=env)
	# optimizers
	transition_optimizer = optax.adam(learning_rate=dynamics_lr)
	# update functions
	transition_update = gradient_update_fn(transition_loss, transition_optimizer, pmap_axis_name=None)

	def wm_training_step(training_state: TrainingState, transitions: dict, key):
		# update environment models
		obs = jnp.array(transitions['obs'])
		next_obs = jnp.array(transitions['obs2'])
		actions = jnp.array(transitions['act'])
		rewards = jnp.array(transitions['rew'])

		tloss, transition_params, transition_optimizer_state, _ = transition_update(training_state.transition_params,
																				training_state.preprocessor_params,
																				obs, actions, next_obs,
																				optimizer_state=training_state.transition_optimizer_state)

		new_train_state = training_state.replace(transition_params=transition_params,
												 transition_optimizer_state=transition_optimizer_state)

		metrics = {'tloss': tloss}

		return new_train_state, metrics
    
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


	# ---
	eval_key, key = jax.random.split(key)
	evaluator = Evaluator(
						eval_env,
						partial(make_policy, deterministic=True),
						episode_length=episode_length,
						action_repeat=action_repeat,
						key=eval_key
						)

	init_training_state_key, eval_key, key = jax.random.split(key, 3)
	training_state = init_training_state(init_training_state_key, obs_size, svg_networks, transition_optimizer)


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

	return make_policy, (svg_networks, training_state),\
	 wm_metrics, replay_buffer




