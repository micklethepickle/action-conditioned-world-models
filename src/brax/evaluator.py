import time
from typing import Callable, Sequence, Tuple

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
from brax.training.acting import actor_step
import jax
import numpy as np


def generate_unroll(
    env: envs.Env,
    env_state: envs.State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[envs.State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, policy, current_key, extra_fields=extra_fields)
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length)
  return final_state, data

class Evaluator:
  """Class to run evaluations."""

  def __init__(self, eval_env: envs.Env,
               eval_policy_fn: Callable[[PolicyParams],
                                        Policy],
               episode_length: int, action_repeat: int, key: PRNGKey):
    """Init.
    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.

    eval_env = envs.wrappers.EvalWrapper(eval_env)

    def generate_eval_unroll(policy_params: PolicyParams,
                             key: PRNGKey) -> envs.State:
      eval_first_state = eval_env.reset(key)
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat)[0]

    self._generate_eval_unroll = jax.jit(generate_eval_unroll)
    self._steps_per_unroll = episode_length

  def run_evaluation(self, policy_params: PolicyParams,
                     training_metrics: Metrics) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state = self._generate_eval_unroll(policy_params, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {
        f'eval/episode_{name}': np.mean(value)
        for name, value in eval_metrics.episode_metrics.items()
    }
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics
    }

    return metrics