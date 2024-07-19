import tqdm
from typing import Any, Callable

import jax
import flax
import jax.numpy as jnp
from functools import partial
import gymnax
import optax
import numpy as np

from src.brax.svginf import losses, networks
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
	reward_optimizer_state: optax.OptState
	reward_params: Any
	transition_optimizer_state: optax.OptState
	transition_params: Any
	critic_params: Any
	target_critic_params: Any
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
	target_critic_params = detach(critic_params)
	critic_optimizer_state = critic_optimizer.init(critic_params)
	
	training_state = TrainingState(
		policy_optimizer_state=policy_optimizer_state,
		policy_params=policy_params,
		reward_optimizer_state=reward_optimizer_state,
		reward_params=reward_params,
		transition_optimizer_state=transition_optimizer_state,
		transition_params=transition_params,
		critic_params=critic_params,
		target_critic_params=target_critic_params,
		critic_optimizer_state=critic_optimizer_state,
		preprocessor_params=None
	)
	return training_state

def train(env: envs.Env,
		eval_env: envs.Env,
		episode_length: int,
		unroll_length: int,
		chunk_length: int,
		num_steps: int,
		warmup_steps: int,
		dynamics_update_every: int,
		policy_update_every: int,
		batch_size: int,
		difference_transition: bool = True,
		true_reward: bool = False,
		true_timesteps: bool = True,
		policy_batch_size: int = 1,
		eval_every: int = 0,
		action_repeat: int = 1,
		seed: int = 0,
		network_sizes=(64,64),
		bp_discount=0.99,
		discount=0.99,
		tau=0.005,
		entropy_init=1000,
		entropy_decay_rate=0.99,
		entropy_transition_steps=500,
		dynamics_lr=0.01,
		policy_lr=0.001,
		critic_lr=0.001,
		grad_clip=10,
		buffer_max=int(1e6),
		bootstrap=1,
		stochastic_transition=False,
		eval_batch_size=1,
		progress_fn: Callable[[int, Any], None] = lambda *args: None,
		with_tqdm=False,
		):

	assert warmup_steps > episode_length
	if eval_every == 0:
		eval_every = max(dynamics_update_every, policy_update_every)
	key = jax.random.PRNGKey(seed)
	env = wrappers.EpisodeWrapper(env, episode_length, action_repeat=action_repeat)
	eval_env = wrappers.EpisodeWrapper(eval_env, episode_length, action_repeat=action_repeat)

	obs_size = env.observation_size
	action_size = env.action_size

	# Make networks and define loss functions
	svg_networks = networks.make_svg_networks(env.observation_size,
											 env.action_size, hidden_layer_sizes=network_sizes,
											 activation=flax.linen.swish,
											 true_timesteps=true_timesteps,
											 difference_transition=difference_transition)
	make_policy = networks.make_inference_fn(svg_networks)
	transition_loss, reward_loss, policy_loss, critic_loss = losses.make_losses(svg_networks, discount=discount, bp_discount=bp_discount,
																   env=env, unroll_length=unroll_length, 
																   make_policy=make_policy,
																   reward_function=env.make_reward_fn() if true_reward else None,
																  policy_batch_size=policy_batch_size,
																  bootstrap=bootstrap,
																  stochastic_transition=stochastic_transition)
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

	# entropy decay
	entropy_reg_fn = optax.exponential_decay(init_value=entropy_init, transition_steps=entropy_transition_steps, decay_rate=entropy_decay_rate)

	def wm_training_step(training_state: TrainingState, transitions: dict, key):
		ckey, key = jax.random.split(key)
		obs = jnp.array(transitions['obs'])
		next_obs = jnp.array(transitions['obs2'])
		actions = jnp.array(transitions['act'])
		dones = jnp.array([transitions['term']])
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

		# update critic
		critic_loss, critic_params, critic_optimizer_state, _ = critic_update(training_state.critic_params,
																			training_state.preprocessor_params,
																			training_state.policy_params,
																			training_state.transition_params,
																			training_state.reward_params,
																			training_state.target_critic_params,
																			obs,
																			dones,
																			ckey,
																			optimizer_state=training_state.critic_optimizer_state)

		# update target critic
		target_critic_params = target_update(critic_params, training_state.target_critic_params, tau)

		new_train_state = training_state.replace(transition_params=transition_params,
												reward_params=reward_params,
												critic_params=critic_params,
												target_critic_params=target_critic_params,
												critic_optimizer_state=critic_optimizer_state,
												reward_optimizer_state=reward_optimizer_state,
												transition_optimizer_state=transition_optimizer_state)

		metrics = {'tloss': tloss, 'rloss': rloss, 'closs': critic_loss}

		return new_train_state, metrics

	def policy_training_step(training_state: TrainingState, transitions: dict, entropy_reg: float, key):
		pkey, key = jax.random.split(key)
		# update policy through imagined trajectories
		sampled_init_obs = jnp.array(transitions['obs'][:, 0:1, :])
		(value, aux), policy_params, policy_optimizer_state, p_grad = policy_update(training_state.policy_params,
																	 training_state.preprocessor_params,
																	 training_state.transition_params,
																	 training_state.reward_params,
																	 training_state.critic_params,
																	 sampled_init_obs,
																	 entropy_reg,
																	 pkey,
																	optimizer_state=training_state.policy_optimizer_state)
		p_grad_norms = optax.global_norm(p_grad)

		new_train_state = training_state.replace(policy_params=policy_params,
												policy_optimizer_state=policy_optimizer_state)

		metrics = {'img_ret': aux['img_ret'],
				   'entropy': aux['entropy'], 'grad_norms': p_grad_norms,
				  'ploss': value}
		return new_train_state, metrics

	def actor_step(env_state: envs.State, preprocessor_params: Any, policy_params: Any, key):
		"""Wrapper for acting.actor_step so it's jittable.
		Taking variable env from outside this scope
		"""
		policy = make_policy((preprocessor_params, policy_params))
		env_state, transition = acting.actor_step(env, env_state, policy, key)
		return env_state, transition

	# Jitting stuff
	jit_train_wm = jax.jit(wm_training_step)
	jit_train_policy = jax.jit(policy_training_step)
	jit_actor_step = jax.jit(actor_step)
	jit_env_reset = jax.jit(env.reset)


	env_reset_key, init_training_state_key, eval_key, key = jax.random.split(key, 4)
	training_state = init_training_state(init_training_state_key, obs_size, svg_networks,
						   reward_optimizer, transition_optimizer, policy_optimizer, critic_optimizer)
	env_state = jit_env_reset(env_reset_key)
	policy = make_policy((training_state.preprocessor_params, training_state.policy_params))

	replay_buffer = SeqReplayBuffer(buffer_max, 
							observation_dim=obs_size,
							action_dim=action_size,
							sampled_seq_len=chunk_length,
							observation_type=np.float64)
	# For evaluation
	evaluator = Evaluator(
						  eval_env,
						  partial(make_policy, deterministic=True),
						  episode_length=episode_length,
						  action_repeat=action_repeat,
						  key=eval_key
						  )

	ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []
	all_returns = []
	all_metrics = []
	metrics = {}
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
			env_state = jit_env_reset(reset_key)
			all_returns.append(np.sum(ep_rewards))
			ep_obs, ep_actions, ep_rewards, ep_terminals, ep_next_obs = [], [], [], [], []
		
		# update
		if i % dynamics_update_every == 0 and i > warmup_steps:	
			train_key, key = jax.random.split(key)
			sampled_episodes = replay_buffer.random_episodes(batch_size)
			training_state, wm_metrics = jit_train_wm(training_state, sampled_episodes, train_key)

		if i % policy_update_every == 0 and i > warmup_steps:
			train_key, key = jax.random.split(key)
			policy_sampled_episodes = replay_buffer.random_episodes(policy_batch_size)
			entropy_reg = entropy_reg_fn(i)
			training_state, policy_metrics = jit_train_policy(training_state, policy_sampled_episodes, entropy_reg, train_key)

		# Run evals
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


	return make_policy, (svg_networks, training_state), all_metrics, replay_buffer

		



