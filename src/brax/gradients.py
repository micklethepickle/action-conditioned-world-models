
"""Brax training gradient utility functions."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax

def clip_by_global_norm(updates):
  g_norm = optax.global_norm(updates)
  trigger = g_norm < max_gradient_norm
  return jax.tree_map(
      lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
      updates)

def loss_and_pgrad(loss_fn: Callable[..., float],
                   pmap_axis_name: Optional[str],
                   has_aux: bool = False):
  g = jax.value_and_grad(loss_fn, has_aux=has_aux)

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h


def gradient_update_fn(loss_fn: Callable[..., float],
                       optimizer: optax.GradientTransformation,
                       pmap_axis_name: Optional[str],
                       has_aux: bool = False,
                       max_gradient_norm=None):
  """Wrapper of the loss function that apply gradient updates.
  Args:
    loss_fn: The loss function.
    optimizer: The optimizer to apply gradients.
    pmap_axis_name: If relevant, the name of the pmap axis to synchronize
      gradients.
    has_aux: Whether the loss_fn has auxiliary data.
  Returns:
    A function that takes the same argument as the loss function plus the
    optimizer state. The output of this function is the loss, the new parameter,
    and the new optimizer state.
  """
  def clip_by_global_norm(updates):
    g_norm = optax.global_norm(updates)
    trigger = g_norm < max_gradient_norm
    updates = jax.tree_map(
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
        updates)
    return jax.tree_map(lambda u: jnp.nan_to_num(u), updates)
  def clip_grad(updates):
    return jax.tree_map(lambda g: jnp.clip(jnp.nan_to_num(g), a_min=-max_gradient_norm, a_max=max_gradient_norm), updates)

  loss_and_pgrad_fn = loss_and_pgrad(
      loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

  def f(*args, optimizer_state):
    value, grads = loss_and_pgrad_fn(*args)
    if max_gradient_norm is not None:
      clipped_grads = clip_by_global_norm(grads)
    else:
      clipped_grads = grads
    params_update, optimizer_state = optimizer.update(clipped_grads, optimizer_state)
    params = optax.apply_updates(args[0], params_update)
    return value, params, optimizer_state, grads

  return f