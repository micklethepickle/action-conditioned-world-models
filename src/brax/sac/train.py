# replay buffer

import jax.numpy as jnp
import jax
import tqdm
from typing import Dict, Any, Callable
import optax
import jax.numpy as jnp
import numpy as np
import random
from collections import deque
from brax import envs


# ALGO LOGIC: initialize agent here:
import flax
from typing import Sequence
import flax.linen as nn
from flax.training.train_state import TrainState
from src.brax import networks
from brax.training import distribution, types
from brax.training import acting
from src.brax.evaluator import Evaluator
from functools import partial
from src.brax.custom_envs import wrappers

class BasicBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class TrainState(TrainState):
    target_params: flax.core.FrozenDict

def make_inference_fn(parametric_action_distribution,
                     policy_network):
    def make_policy(params,
                  deterministic: bool = False,
                  get_dist: bool  =  False):

        def policy(observations: types.Observation,
                   key_sample):
            logits = policy_network.apply(params, observations)
            if deterministic:
                mode = parametric_action_distribution.create_dist(logits).loc
                return parametric_action_distribution.postprocess(mode), {}

            if not get_dist:
                return parametric_action_distribution.sample(
                  logits, key_sample), {'entropy': parametric_action_distribution.entropy(logits, key_sample)}
            else:
                dist = parametric_action_distribution.create_dist(logits)
                return parametric_action_distribution.sample(
                  logits, key_sample), {'entropy': parametric_action_distribution.entropy(logits, key_sample),
                                    'loc': dist.loc,
                                    'scale': dist.scale}


        return policy

    return make_policy


def train(
        env: envs.Env,
        eval_env: envs.Env,
        episode_length: int,
        num_steps: int,
        warmup_steps: int,
        dynamics_update_every: int,
        policy_update_every: int,
        batch_size: int,
        eval_every: int = 10,
        action_repeat: int = 1,
        seed: int = 0,
        network_sizes=(64,64),
        discount=0.99,
        entropy_init=0.01,
        lr=3e-3,
        tau=0.005,
        buffer_max=int(1e6),
        progress_fn: Callable[[int, Any], None] = lambda *args: None,
        with_tqdm=False
        ):
    # entropy_init = entropy_init/episode_length # normalize
    key = jax.random.PRNGKey(seed)
    eval_env = wrappers.EpisodeWrapper(eval_env, episode_length, action_repeat=1)
    # actor
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=env.action_size)
    actor = networks.make_policy_network(
            parametric_action_distribution.param_size,
            env.observation_size,
            preprocess_observations_fn=None,
            hidden_layer_sizes=(256, 256),
            activation=nn.relu
    )
    make_policy = make_inference_fn(parametric_action_distribution, actor)


    sample_obs = env.reset(key).obs
    sample_act = jnp.zeros(env.action_size)

    actor_key, qf1_key, key = jax.random.split(key, 3)

    qf1 = QNetwork()
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key),
        target_params=actor.init(actor_key),
        tx=optax.adam(learning_rate=lr),
    )
    qf1_state = TrainState.create(
        apply_fn=qf1.apply,
        params=qf1.init(qf1_key, sample_obs, sample_act),
        target_params=qf1.init(qf1_key, sample_obs, sample_act),
        tx=optax.adam(learning_rate=lr),
    )
    actor.apply = jax.jit(actor.apply)
    qf1.apply = jax.jit(qf1.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        key
    ):
        target_policy = make_policy(actor_state.target_params)
        next_state_actions, extra = target_policy(next_observations, key)
        
        qf1_next_target = qf1.apply(qf1_state.target_params, next_observations, next_state_actions).reshape(-1)
        # print(extra['entropy'].shape, qf1_next_target.shape)
        qf1_next_target += entropy_init * extra['entropy']
    #     print(qf1_next_target.shape, rewards.shape, dones.shape)
        next_q_value = (rewards + (1 - dones) * discount * (qf1_next_target)).reshape(-1)

        def mse_loss(params):
            qf1_a_values = qf1.apply(params, observations, actions).squeeze()
            return ((qf1_a_values - next_q_value) ** 2).mean(), qf1_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads)
        return qf1_state, qf1_loss_value, qf1_a_values

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
        key
    ): 
        def actor_loss(params):
            policy = make_policy(params)
            action, extra = policy(observations, key)
            return -(qf1.apply(qf1_state.params, observations, action).mean() + (entropy_init * extra['entropy']).mean())

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(actor_state.params, actor_state.target_params, tau)
        )
        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, tau)
        )
        return actor_state, qf1_state, actor_loss_value

    @jax.jit
    def actor_step(env_state: Any, policy_params: Any, key):
        policy = make_policy(policy_params)
        env_state, transition = acting.actor_step(env, env_state, policy, key)
        return env_state, transition


    eval_key, key = jax.random.split(key)
    evaluator = Evaluator(
        eval_env,
        partial(make_policy, deterministic=True),
        episode_length=episode_length,
        action_repeat=1,
        key=eval_key
    )

    rb = BasicBuffer(max_size=buffer_max)
    jit_reset = jax.jit(env.reset)
    env_state = jit_reset(key)
    all_metrics = []
    iterator = tqdm.tqdm(range(num_steps)) if with_tqdm else range(num_steps)
    for i in iterator:
        # get action (might be exploration?) and take step
        step_key, key = jax.random.split(key)
        env_state, transition = actor_step(env_state, actor_state.params, step_key)

        
        # save transition in rb and handle terminal states
        rb.push(transition.observation, transition.action, transition.reward,
                transition.next_observation, env_state.done)
        
        if env_state.done:
            reset_key, key = jax.random.split(key)
            env_state = jit_reset(reset_key)
        
        
        # training
        if i > warmup_steps:
            # sample from rb
            observations, actions, rewards, next_observations, dones = rb.sample(batch_size)
            observations = jnp.array(observations)
            actions = jnp.array(actions)
            rewards = jnp.array(rewards).reshape(-1)
            next_observations = jnp.array(next_observations)
            dones = jnp.array(dones).reshape(-1)
            
            if i % dynamics_update_every == 0:
                critic_key, actor_key = jax.random.split(key)
                qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                    actor_state,
                    qf1_state,
                    observations,
                    actions,
                    next_observations,
                    rewards,
                    dones,
                    critic_key
                )
            
            if i % policy_update_every == 0:
                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    observations,
                    actor_key
                )
            
            if i % eval_every == 0:
                eval_metrics = evaluator.run_evaluation(actor_state.params, {'critic_loss': qf1_loss_value, 'actor_loss': actor_loss_value})
                all_metrics.append(eval_metrics)
                progress_fn(i, eval_metrics)

    return make_policy, (actor_state, qf1_state), all_metrics, rb
