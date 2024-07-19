from jax import numpy as jp
from brax.envs import env as brax_env 


class EpisodeWrapper(brax_env.Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: brax_env.Env, episode_length: int,
               action_repeat: int):
    super().__init__(env)
    if hasattr(self, 'unwrapped'):
      if hasattr(self.unwrapped, 'sys'):
        self.unwrapped.sys.config.dt *= action_repeat
        self.unwrapped.sys.config.substeps *= action_repeat
    else:
      self.dt *= action_repeat
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(())
    state.info['truncation'] = jp.zeros(())
    return state

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    state = self.env.step(state, action)
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    done = jp.where(steps >= self.episode_length, one, state.done)
    state.info['truncation'] = jp.where(steps >= self.episode_length,
                                        1 - state.done, zero)
    state.info['steps'] = steps
    return state.replace(done=done)