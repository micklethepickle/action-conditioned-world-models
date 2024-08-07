import jax.numpy as jnp

from typing import Optional, Union

from src.brax.custom_envs.myriad.custom_types import Params
from src.brax.custom_envs.myriad.base import IndirectFHCS

class PredatorPrey(IndirectFHCS):
  # TODO: there is an error when trying to plot with PredatorPrey
  """
    Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 22, Lab 13)
    The states evolution is base on a standard Lotka-Volterra model.
    This particular environment is inspired from Bean San Goh, George Leitmann, and Thomas L. Vincent.
    Optimal control of a prey-predator system. Mathematical Biosciences, 19, 1974.

    This environment models the evolution of a pest (prey) population ( \\(x_0(t)\\) ) and a predator population ( \\(x_1(t) \\)) in
    the presence of a pesticide ( \\(u(t)\\) ) that affects both the pest and predator populations. The objective in mind is
    to minimize the final pest population, while limiting the usage of the pesticide. Thus:

    .. math::

      \\begin{align}
      & \\min_{u} \\quad && x_0(T) + \\frac{A}{2}\\int_0^T u(t)^2 dt \\\\
      & \\; \\mathrm{s.t.}\\quad && x_0'(t) = (1 - x_1(t))x_0(t) - d_1x_0(t)u(t) \\\\
      & && x_1'(t) = (x_0(t) - 1)x_1(t) - d_2x_1(t)(t)u(t) \\\\
      & && 0 \\leq u(t) \\leq M, \\quad \\int_0^T u(t) dt = B
      \\end{align}

    The particularity here is that the total amount of pesticide to be applied is fixed. To take into account this
    constraint, a virtual state variable ( \\(z(t)\\) ) is added where:

    .. math::

      z'(t) = u(t), \\; z(0) = 0, \\; z(T) = B

    Finally, note that `guess_a` and `guess_b` have been carefully chosen in the study cases to allow for fast iteration
    and ensure convergence.

    Notes
    -----
    x_0: Initial density of the pest and prey population \\( (x_0, x_1) \\)
  """

  def __init__(self, d_1=.1, d_2=.1, A=1., B=5.,
               guess_a=-.52, guess_b=.5, M=1.,
               x_0=(10., 1., 0.), T=100.):
    super().__init__(
      x_0=jnp.array([
        x_0[0],
        x_0[1],
        x_0[2]
      ]),  # Starting state
      x_T=[None, None, B],  # Terminal state, if any
      T=T,  # Duration of experiment
      bounds=jnp.array([  # Bounds over the states (x_0, x_1 ...) are given first,
        [0., 11.],  # followed by bounds over controls (u_0, u_1, ...)
        [0., 11.],
        [0., 5.],
        [0, M]
      ]),
      terminal_cost=True,
      discrete=False,
    )

    self.adj_T = jnp.array([1, 0, 0])  # Final condition over the adjoint, if any
    self.d_1 = d_1
    """Impact of the pesticide on the pest population"""
    self.d_2 = d_2
    """Impact of the pesticide on the prey population"""
    self.A = A
    """Weight parameter balancing the cost"""
    self.guess_a = guess_a
    """Node 2 at which the secant method begins its iteration (Newton's method)"""
    self.guess_b = guess_b
    """Node 1 at which the secant method begins its iteration (Newton's method)"""
    self.M = M
    """Bound on pesticide application at a given time"""

  def dynamics(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
               v_t: Optional[Union[float, jnp.ndarray]] = None, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    x_0, x_1, x_2 = x_t
    if u_t.ndim > 0:
      u_t, = u_t

    d_x = jnp.array([
      (1 - x_1) * x_0 - self.d_1 * x_0 * u_t,
      (x_0 - 1) * x_1 - self.d_2 * x_1 * u_t,
      u_t,
    ])

    return d_x

  def parametrized_dynamics(self, params: Params, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                            v_t: Optional[Union[float, jnp.ndarray]] = None,
                            t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    d_1 = params['d_1']
    d_2 = params['d_2']
    x_0, x_1, x_2 = x_t
    if u_t.ndim > 0:
      u_t, = u_t

    d_x = jnp.array([
      (1 - x_1) * x_0 - d_1 * x_0 * u_t,
      (x_0 - 1) * x_1 - d_2 * x_1 * u_t,
      u_t,
    ])

    return d_x

  def cost(self, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray], t: Optional[jnp.ndarray] = None) -> float:
    return self.A * 0.5 * u_t ** 2

  def parametrized_cost(self, params: Params, x_t: jnp.ndarray, u_t: Union[float, jnp.ndarray],
                        t: Optional[jnp.ndarray] = None) -> float:
    return self.A * 0.5 * u_t ** 2  # Not learning cost for now

  def terminal_cost_fn(self, x_T: Optional[jnp.ndarray], u_T: Optional[jnp.ndarray],
                       T: Optional[jnp.ndarray] = None) -> float:
    return x_T[0]

  def adj_ODE(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray], u_t: Optional[jnp.ndarray],
              t: Optional[jnp.ndarray]) -> jnp.ndarray:
    return jnp.array([
      adj_t[0] * (x_t[1] - 1 + self.d_1 * u_t[0]) - adj_t[1] * x_t[1],
      adj_t[0] * x_t[0] + adj_t[1] * (1 - x_t[0] + self.d_2 * u_t[0]),
      0
    ])

  def optim_characterization(self, adj_t: jnp.ndarray, x_t: Optional[jnp.ndarray],
                             t: Optional[jnp.ndarray]) -> jnp.ndarray:
    char = (adj_t[:, 0] * self.d_1 * x_t[:, 0] + adj_t[:, 1] * self.d_2 * x_t[:, 1] - adj_t[:, 2]) / self.A
    char = char.reshape(-1, 1)

    return jnp.minimum(self.bounds[-1, 1], jnp.maximum(self.bounds[-1, 0], char))