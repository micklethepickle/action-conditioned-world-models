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

from src.brax.offline_svginf import losses, networks
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
		num_steps: int,
		policy_steps: int,
		batch_size: int,
		true_reward: bool = False,
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
	env = wrappers.EpisodeWrapper(env, episode_length, action_repeat=action_repeat)
	eval_env = wrappers.EpisodeWrapper(eval_env, episode_length, action_repeat=action_repeat)

	obs_size = env.observation_size
	action_size = env.action_size

	# Make networks and define loss functions
	svg_networks = networks.make_svg_networks(env.observation_size,
											 env.action_size, hidden_layer_sizes=network_sizes,
											 activation=flax.linen.swish)
	make_policy = networks.make_inference_fn(svg_networks)
	transition_loss, reward_loss, policy_loss, critic_loss = losses.make_losses(svg_networks, discount=discount, bp_discount=bp_discount,
																   env=env, unroll_length=unroll_length, 
																   make_policy=make_policy,
																   reward_function=env.make_reward_fn() if true_reward else None,
																  policy_batch_size=policy_batch_size,
																  bootstrap=bootstrap)
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

	def policy_training_step(training_state: TrainingState, transitions: dict, entropy_reg: float, key):
		pkey, ckey = jax.random.split(key)
		sampled_init_obs = jnp.array(transitions['obs'][:, 0:1, :])
		(value, aux), policy_params, policy_optimizer_state, p_grad_norms = policy_update(training_state.policy_params,
															 training_state.preprocessor_params,
															 training_state.transition_params,
															 training_state.reward_params,
															 training_state.critic_params,
															 sampled_init_obs,
															 entropy_reg,
															 pkey,
															optimizer_state=training_state.policy_optimizer_state)

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

		metrics = {'img_ret': aux['img_ret'],
				   'entropy': aux['entropy'], 'grad_norms': optax.global_norm(p_grad_norms),
				  'ploss': value, 'closs': critic_loss}

		return new_train_state, metrics

	# Jitting stuff
	jit_wm_training_step = jax.jit(wm_training_step)
	jit_train_policy = jax.jit(policy_training_step)


	init_training_state_key, eval_key, key = jax.random.split(key, 3)
	training_state = init_training_state(init_training_state_key, obs_size, svg_networks,
						   reward_optimizer, transition_optimizer, policy_optimizer, critic_optimizer)

	# For evaluation
	evaluator = Evaluator(
						  eval_env,
						  partial(make_policy, deterministic=True),
						  episode_length=episode_length,
						  action_repeat=action_repeat,
						  key=eval_key
						  )

	print('Filling dataset with random* trajectories')
	key, data_key = jax.random.split(key)
	smearing_rates = [0., 0.5]
	replay_buffer = get_dataset(env, episode_length, smearing_rates, buffer_max, data_key)

	wm_metrics = []
	iterator = tqdm.tqdm(range(num_steps)) if with_tqdm else range(num_steps)
	print('Training world model')
	for i in iterator:
		
		train_key, key = jax.random.split(key)
		sampled_episodes = replay_buffer.random_episodes(batch_size)
		training_state, wm_met = jit_wm_training_step(training_state, sampled_episodes, train_key)
		wm_metrics.append(wm_met)
		progress_fn(i, wm_met)

	policy_metrics = []
	print('Training policy')
	p_itr = tqdm.tqdm(range(policy_steps)) if with_tqdm else range(policy_steps)
	for i in p_itr:
		train_key, key = jax.random.split(key)

		policy_samples = replay_buffer.random_episodes(policy_batch_size)
		entropy_reg = entropy_reg_fn(i)
		training_state, policy_met = jit_train_policy(training_state, policy_samples, entropy_reg, train_key)

		# Run evals
		if i % eval_every == 0:
			eval_metrics = evaluator.run_evaluation((training_state.preprocessor_params, training_state.policy_params), policy_met)
			policy_metrics.append(eval_metrics)
			progress_fn(i, eval_metrics)

	transition_network = MultistepTransition(svg_networks.transition_network)
	return make_policy, (training_state.preprocessor_params, training_state.policy_params), (transition_network, training_state), wm_metrics, policy_metrics, replay_buffer



