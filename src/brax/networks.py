import dataclasses
from typing import Any, Callable, Sequence, Tuple, Optional, Mapping, Dict
from brax.training import types
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen
from flax.linen.linear import default_kernel_init

from src.trajectory_flax_gpt2 import FlaxGPT2Model
from transformers import GPT2Config
from src.brax.s4 import BatchStackedModel, S4Layer
from functools import partial

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any	# this could be a real type?
Array = Any
Carry = Any
CarryHistory = Any
Output = Any

@dataclasses.dataclass
class FeedForwardNetwork:
	init: Callable[..., Any]
	apply: Callable[..., Any]

@dataclasses.dataclass
class SequenceModel:
	init: Callable[..., Any]
	apply_sequence: Callable[..., Any]
	apply_recurrence: Callable[..., Any]
	prime_recurrence: Callable[..., Any]
	apply_sequence_extra: Callable[..., Any] = None

class Embed(linen.Module):
	embd_size: int
	kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
	bias: bool = True

	@linen.compact
	def __call__(self, data: jnp.ndarray):
		embding	= linen.Dense(self.embd_size, kernel_init=self.kernel_init,
					use_bias=self.bias)(data)
		return embding

class MLP(linen.Module):
	"""MLP module."""
	layer_sizes: Sequence[int]
	activation: ActivationFn = linen.relu
	kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
	activate_final: bool = False
	bias: bool = True

	@linen.compact
	def __call__(self, data: jnp.ndarray):
		hidden = data
		for i, hidden_size in enumerate(self.layer_sizes):
			hidden = linen.Dense(
					hidden_size,
					name=f'hidden_{i}',
					kernel_init=self.kernel_init,
					use_bias=self.bias)(
							hidden)
			if i != len(self.layer_sizes) - 1 or self.activate_final:
				hidden = self.activation(hidden)
		return hidden

class NormMLP(linen.Module):
	"""MLP module."""
	layer_sizes: Sequence[int]
	min_max_range: float = 40
	activation: ActivationFn = linen.relu
	kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
	activate_final: bool = False
	bias: bool = True

	@linen.compact
	def __call__(self, data: jnp.ndarray):
		hidden = data
		for i, hidden_size in enumerate(self.layer_sizes):
			hidden = linen.Dense(
					hidden_size,
					name=f'hidden_{i}',
					kernel_init=self.kernel_init,
					use_bias=self.bias)(
							hidden)
			if i != len(self.layer_sizes) - 1 or self.activate_final:
				hidden = self.activation(hidden)
		hidden = (linen.sigmoid(hidden) * self.min_max_range) - (self.min_max_range /2)
		return hidden


class DenseParams(linen.Module):
	"""Dummy module for creating parameters matching `flax.linen.Dense`."""

	features: int
	use_bias: bool = True
	param_dtype: Dtype = jnp.float32
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

	@linen.compact
	def __call__(self, inputs: Array) -> Tuple[Array, Optional[Array]]:
		k = self.param(
				'kernel', self.kernel_init, (inputs.shape[-1], self.features),
				self.param_dtype)
		if self.use_bias:
			b = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
		else:
			b = None
		return k, b

class RNNCell(linen.recurrent.RNNCellBase):
	activation_fn: Callable[..., Any] = linen.activation.tanh
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
	recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.orthogonal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
	dtype: Optional[Dtype] = None
	param_dtype: Dtype = jnp.float32

	@linen.compact
	def __call__(self, h, inputs):
		"""An optimized long short-term memory (LSTM) cell.

		Args:
			carry: the hidden state of the LSTM cell, initialized using
				`LSTMCell.initialize_carry`.
			inputs: an ndarray with the input for the current time step. All
				dimensions except the final are considered batch dimensions.

		Returns:
			A tuple with the new carry and the output.
		"""
		hidden_features = h.shape[-1]

		def _concat_dense(inputs: Array,
						params: Mapping[str, Tuple[Array, Optional[Array]]],
						use_bias: bool = True) -> Dict[str, Array]:
			# Concatenates the individual kernels and biases, given in params, into a
			# single kernel and single bias for efficiency before applying them using
			# dot_general.
			kernels = [kernel for kernel, _ in params.values()]
			kernel = jnp.concatenate(kernels, axis=-1)
			if use_bias:
				biases = []
				for _, bias in params.values():
					if bias is None:
						raise ValueError('bias is None but use_bias is True.')
					biases.append(bias)
				bias = jnp.concatenate(biases, axis=-1)
			else:
				bias = None
			inputs, kernel, bias = linen.dtypes.promote_dtype(inputs, kernel, bias, dtype=self.dtype)
			y = jnp.dot(inputs, kernel)
			if use_bias:
				# This assert is here since mypy can't infer that bias cannot be None
				assert bias is not None
				y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

			# Split the result back into individual (i, f, g, o) outputs.
			split_indices = np.cumsum([kernel.shape[-1] for kernel in kernels[:-1]])
			ys = jnp.split(y, split_indices, axis=-1)
			return dict(zip(params.keys(), ys))

		# Create params with the same names/shapes as `LSTMCell` for compatibility.
		dense_params_h = {}
		dense_params_i = {}
		dense_params_i['h'] = DenseParams(
			features=hidden_features, use_bias=True,
			param_dtype=self.param_dtype,
			kernel_init=self.kernel_init, bias_init=self.bias_init,
			name=f'ih')(inputs)
		dense_params_h['h'] = DenseParams(
			features=hidden_features, use_bias=True,
			param_dtype=self.param_dtype,
			kernel_init=self.recurrent_kernel_init, bias_init=self.bias_init,
			name=f'hh')(h)

		dense_h = _concat_dense(h, dense_params_h, use_bias=True)
		dense_i = _concat_dense(inputs, dense_params_i, use_bias=False)

		new_h = self.activation_fn(dense_h['h'] + dense_i['h'])

		return new_h, new_h
	@staticmethod
	def initialize_carry(rng, batch_dims, size, init_fn=jax.nn.initializers.zeros):
		c_shape = batch_dims + (size, )
		key1, rng = jax.random.split(rng)

		return init_fn(key1, c_shape)

class SimpleLSTM(linen.Module):
	@linen.compact
	def __call__(self, c, xs):
		LSTM = linen.scan(linen.OptimizedLSTMCell,
							variable_broadcast="params",
								split_rngs={"params": False},
								in_axes=1,
								out_axes=1)
			
		return LSTM()(c, xs)

class SimpleRNN(linen.Module):
	@linen.compact
	def __call__(eslf, c, xs):
		RNN = linen.scan(RNNCell,
						variable_broadcast='params',
						split_rngs={'params': False},
						in_axes=1,
						out_axes=1)

		return RNN()(c, xs)


def make_policy_network(
		param_size: int,
		obs_size: int,
		preprocess_observations_fn: types.PreprocessObservationFn = types
		.identity_observation_preprocessor,
		hidden_layer_sizes: Sequence[int] = (256, 256),
		activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
	"""Creates a policy network."""
	policy_module = MLP(
			layer_sizes=list(hidden_layer_sizes) + [param_size],
			activation=activation,
			kernel_init=jax.nn.initializers.lecun_uniform())

	def apply(processor_params, policy_params, obs):
		obs = preprocess_observations_fn(obs, processor_params)
		return policy_module.apply(policy_params, obs)

	def apply_no_preprocess(policy_params, obs):
		return policy_module.apply(policy_params, obs)

	dummy_obs = jnp.zeros((1, obs_size))

	if preprocess_observations_fn is None:
		return FeedForwardNetwork(
				init=lambda key: policy_module.init(key, dummy_obs), apply=apply_no_preprocess)
	else:
		return FeedForwardNetwork(
				init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_transition_network(
	obs_size: int,
	action_size: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types
	.identity_observation_preprocessor,
	hidden_layer_sizes: Sequence[int] = (256, 256),
	activation: ActivationFn = linen.relu,
	true_timesteps: bool = False,
	difference_transition: bool = True) -> FeedForwardNetwork:
	"""Creates transition network"""
	out_size = obs_size - 1 if true_timesteps else obs_size
	transition_module = MLP(
		layer_sizes=list(hidden_layer_sizes) + [out_size],
		activation=activation,
		kernel_init=jax.nn.initializers.lecun_uniform())

	def apply(processor_params, transition_params, obs, actions):
		obs = preprocess_observations_fn(obs, processor_params)
		next_obs = transition_module.apply(transition_params, jnp.concatenate((obs, actions), axis=-1))
		if true_timesteps:
			next_timesteps = obs[..., -1:] + 1.
		if difference_transition:
			if true_timesteps:
				next_obs = obs[..., :-1] + next_obs
				next_obs = jnp.concatenate((next_obs, next_timesteps), axis=-1)
			else:
				next_obs = obs + next_obs
		elif true_timesteps:
			next_obs = jnp.concatenate((next_obs, next_timesteps), axis=-1)
		return next_obs

	dummy_obs = jnp.zeros((1, obs_size))
	dummy_action = jnp.zeros((1, action_size))
	return FeedForwardNetwork(
		init=lambda key: transition_module.init(key, jnp.concatenate((dummy_obs, dummy_action), axis=-1)), apply=apply)

def make_gpt_transition_network(
	obs_size: int,
	action_size: int,
	max_episode_length: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types
	.identity_observation_preprocessor,
	decoder_hidden_layer_sizes: Sequence[int] = (32, 32),
	embd_dim: int = 72,
	transformer_nlayers: int = 3,
	transformer_nheads: int = 3,
	transformer_pdrop: float = 0.1,
	true_timesteps: bool = False,
	seed: int = 0) -> FeedForwardNetwork:
	"""Creates Transition network with a GPT core"""
	# Transformer definition
	config = GPT2Config(vocab_size=1,
						n_layer=transformer_nlayers,
						n_head=transformer_nheads,
						n_embd=embd_dim,
						attn_pdrop=transformer_pdrop,
						resid_pdrop=transformer_pdrop,
						embd_pdrop=transformer_pdrop)
	transformer = FlaxGPT2Model(config, seed=seed)

	# Embedders
	init_obs_embder = Embed(embd_size=embd_dim, 
							kernel_init=jax.nn.initializers.lecun_uniform())
	action_embder = Embed(embd_size=embd_dim,
							kernel_init=jax.nn.initializers.lecun_uniform())
	timestep_embder = linen.Embed(max_episode_length + 1, embd_dim, embedding_init=jax.nn.initializers.lecun_uniform())

	# Decoder
	out_size = obs_size - 1 if true_timesteps else obs_size
	decoder = MLP(
		layer_sizes=list(decoder_hidden_layer_sizes) + [out_size],
		activation=linen.relu,
		kernel_init=jax.nn.initializers.lecun_uniform())

	def init(key):
		dummy_obs = jnp.zeros((1, obs_size))
		dummy_action = jnp.zeros((1, action_size))
		dummy_timestep = jnp.arange(0, 1)
		dummy_embd = jnp.zeros((1, embd_dim))

		obs_key, action_key, timestep_key, decoder_key, transformer_key = jax.random.split(key, 5)

		transformer_params = transformer.init_weights(transformer_key, dummy_embd.shape)
		init_obs_embder_params = init_obs_embder.init(obs_key, dummy_obs)
		action_embder_params = action_embder.init(action_key, dummy_action)
		timestep_embder_params = timestep_embder.init(timestep_key, dummy_timestep)
		decoder_params = decoder.init(decoder_key, dummy_embd)

		transition_params = {'transformer_params': transformer_params, 'init_obs_embder_params': init_obs_embder_params,
							'action_embder_params': action_embder_params, 'timestep_embder_params': timestep_embder_params,
							'decoder_params': decoder_params}
		return transition_params

	def apply_sequence(processor_params, params, init_obs, actions, key=None, train=True):
		"""
		init_obs of shape (B, 1, O)
		actions of shape	(B, L, A)

		returns observations of shape (B, L+1, O)
		"""
		sequence_length = actions.shape[1] + 1 # actions length plus the initial obs
		batch_size = actions.shape[0]
		init_obs = preprocess_observations_fn(init_obs, processor_params)
		timesteps = jnp.arange(sequence_length)
		init_obs_embds = init_obs_embder.apply(params['init_obs_embder_params'], init_obs) # (B, 1, H)

		action_embds = action_embder.apply(params['action_embder_params'], actions) # (B, L, H)
		sequence = jnp.concatenate((init_obs_embds, action_embds), axis=1)
		time_embds = jnp.expand_dims(timestep_embder.apply(params['timestep_embder_params'], timesteps), axis=0)
		input_embds = sequence + time_embds

		out_hidden_states = transformer(input_embds, params=params['transformer_params'], past_key_values=None, output_attentions=False)['last_hidden_state']
		obs_preds = decoder.apply(params['decoder_params'], out_hidden_states)
		
		if true_timesteps:
			next_timesteps = jnp.repeat(jnp.arange(sequence_length)[None,...,None], repeats=batch_size, axis=0)
			init_times = init_obs[..., -1:]
			next_timesteps = next_timesteps + init_times
			obs_preds = jnp.concatenate((obs_preds, next_timesteps), axis=-1)
		return obs_preds

	def apply_sequence_extra(processor_params, params, init_obs, actions, key=None, train=True):
		"""
		init_obs of shape (B, 1, O)
		actions of shape	(B, L, A)

		returns observations of shape (B, L+1, O)
		"""
		sequence_length = actions.shape[1] + 1 # actions length plus the initial obs
		batch_size = actions.shape[0]
		init_obs = preprocess_observations_fn(init_obs, processor_params)
		timesteps = jnp.arange(sequence_length)
		init_obs_embds = init_obs_embder.apply(params['init_obs_embder_params'], init_obs) # (B, 1, H)

		action_embds = action_embder.apply(params['action_embder_params'], actions) # (B, L, H)
		sequence = jnp.concatenate((init_obs_embds, action_embds), axis=1)
		time_embds = jnp.expand_dims(timestep_embder.apply(params['timestep_embder_params'], timesteps), axis=0)
		input_embds = sequence + time_embds

		out = transformer(input_embds, params=params['transformer_params'], past_key_values=None, output_attentions=True)
		obs_preds = decoder.apply(params['decoder_params'], out['last_hidden_state'])
		if true_timesteps:
			next_timesteps = jnp.repeat(jnp.arange(sequence_length)[None,...,None], repeats=batch_size, axis=0)
			init_times = init_obs[..., -1:]
			next_timesteps = next_timesteps + init_times
			obs_preds = jnp.concatenate((obs_preds, next_timesteps), axis=-1)
		return obs_preds, out

	def apply_recurrence(processor_params, params, actions, timesteps, cache, key=None, train=True):
		"""
		action of shape (B, 1, A)
		timesteps of shape (1,) - current timestep of the sequence
		cache is recurrent 

		returns next_obs (B, 1, O), and cache
		"""
		batch_size = actions.shape[0]
		action_embds = action_embder.apply(params['action_embder_params'], actions) # (B, 1, H)
		sequence = action_embds
		time_embds = jnp.expand_dims(timestep_embder.apply(params['timestep_embder_params'], timesteps), axis=0)
		input_embds = sequence + time_embds

		out = transformer(input_embds, params=params['transformer_params'], past_key_values=cache)
		cache = out['past_key_values']
		# pkv = None if past_key_values is None else out['past_key_values']
		obs_preds = decoder.apply(params['decoder_params'], out['last_hidden_state'])

		if true_timesteps:
			next_timesteps = timesteps + 1.
			next_timesteps = jnp.repeat(next_timesteps[None, ..., None], repeats=batch_size, axis=0)
			obs_preds = jnp.concatenate((obs_preds, next_timesteps), axis=-1)

		return obs_preds, cache

	def prime_recurrence(processor_params, params, batch_size, unroll_length, init_obs, key, train=True):
		"""
		Primes the recurrence with a given initial observation
		init_obs of shape (B, 1, O)

		return cache necessary for apply_recurrence
		"""
		init_pkv = transformer.init_cache(batch_size, unroll_length)

		init_obs = preprocess_observations_fn(init_obs, processor_params)
		init_obs_embds = init_obs_embder.apply(params['init_obs_embder_params'], init_obs) # (B, 1, H)

		timesteps = jnp.arange(1)
		time_embds = jnp.expand_dims(timestep_embder.apply(params['timestep_embder_params'], timesteps), axis=0)
		input_embds = init_obs_embds + time_embds

		out = transformer(input_embds, params=params['transformer_params'], past_key_values=init_pkv)
		cache = out['past_key_values']
		return cache

	return SequenceModel(init=init, 
						apply_sequence=apply_sequence,
						apply_sequence_extra=apply_sequence_extra,
						apply_recurrence=apply_recurrence, 
						prime_recurrence=prime_recurrence)
	# return FeedForwardNetwork(init=init, apply=apply), prime_recurrence

def make_lstm_transition_network(
	obs_size: int,
	action_size: int,
	max_episode_length: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types
	.identity_observation_preprocessor,
	decoder_hidden_layer_sizes: Sequence[int] = (32, 32),
	embd_dim: int = 72,
	true_timesteps: bool = False,
	seed: int = 0) -> FeedForwardNetwork:

	lstm = SimpleLSTM()

	# Embedders
	# init_obs_embder = Embed(embd_size=embd_dim, 
	# 						kernel_init=jax.nn.initializers.lecun_uniform())
	init_obs_embder = MLP(layer_sizes=list(decoder_hidden_layer_sizes) + [embd_dim],
						activation=linen.relu,
						kernel_init=jax.nn.initializers.lecun_uniform())
	# action_embder = Embed(embd_size=embd_dim,
	# 						kernel_init=jax.nn.initializers.lecun_uniform())
	action_embder = MLP(layer_sizes=list(decoder_hidden_layer_sizes) + [embd_dim],
						activation=linen.relu,
						kernel_init=jax.nn.initializers.lecun_uniform())

	# Decoder
	out_size = obs_size - 1 if true_timesteps else obs_size
	decoder = MLP(
		layer_sizes=list(decoder_hidden_layer_sizes) + [out_size],
		activation=linen.relu,
		kernel_init=jax.nn.initializers.lecun_uniform())

	def init(key):
		dummy_obs = jnp.zeros((1, obs_size))
		dummy_action = jnp.zeros((1, action_size))
		dummy_embd = jnp.zeros((1, embd_dim))
		dummy_sequence = jnp.zeros((1, 1, embd_dim))

		obs_key, action_key, timestep_key, decoder_key, lstm_key = jax.random.split(key, 5)

		init_carry = linen.LSTMCell.initialize_carry(lstm_key, (1,), embd_dim)
		lstm_params = lstm.init(lstm_key, init_carry, dummy_sequence)
		init_obs_embder_params = init_obs_embder.init(obs_key, dummy_obs)
		action_embder_params = action_embder.init(action_key, dummy_action)
		decoder_params = decoder.init(decoder_key, dummy_embd)

		transition_params = {'lstm_params': lstm_params, 'init_obs_embder_params': init_obs_embder_params,
							'action_embder_params': action_embder_params, 
							'decoder_params': decoder_params}
		return transition_params

	def apply_sequence(processor_params, params, init_obs, actions, key=None, train=True):
		"""
		init_obs of shape (B, 1, O)
		actions of shape	(B, L, A)

		returns observations of shape (B, L+1, O)
		"""
		batch_size = actions.shape[0]
		carry = linen.LSTMCell.initialize_carry(key, (batch_size, ), embd_dim)
		init_obs = preprocess_observations_fn(init_obs, processor_params)
		init_obs_embds = init_obs_embder.apply(params['init_obs_embder_params'], init_obs)

		action_embds = action_embder.apply(params['action_embder_params'], actions) # (B, L, H)
		sequence = jnp.concatenate((init_obs_embds, action_embds), axis=1)

		_, out_val = lstm.apply(params['lstm_params'], carry, sequence) # ((B, H), (B, H)), (B, L, H)
		obs_preds = decoder.apply(params['decoder_params'], out_val)

		if true_timesteps:
			sequence_length = actions.shape[1] + 1
			next_timesteps = jnp.repeat(jnp.arange(sequence_length)[None,...,None], repeats=batch_size, axis=0)
			init_times = init_obs[..., -1:]
			next_timesteps = next_timesteps + init_times
			obs_preds = jnp.concatenate((obs_preds, next_timesteps), axis=-1)
		return obs_preds

	def apply_recurrence(processor_params, params, actions, timesteps, cache, key=None, train=True):
		"""
		action of shape (B, 1, A)
		timesteps of shape (1,) - current timestep of the sequence
		cache is recurrent 

		returns next_obs (B, 1, O), and cache
		"""
		carry = cache
		action_embds = action_embder.apply(params['action_embder_params'], actions)
		out_carry, out_val = lstm.apply(params['lstm_params'], carry, action_embds) # ((B, H), (B, H)), (B, L, H)
		obs_preds = decoder.apply(params['decoder_params'], out_val)

		if true_timesteps:
			batch_size = actions.shape[0]
			next_timesteps = timesteps + 1.
			next_timesteps = jnp.repeat(next_timesteps[None, ..., None], repeats=batch_size, axis=0)
			obs_preds = jnp.concatenate((obs_preds, next_timesteps), axis=-1)
		return obs_preds, out_carry

	def prime_recurrence(processor_params, params, batch_size, unroll_length, init_obs, key, train=True):
		"""
		Primes the recurrence with a given initial observation
		init_obs of shape (B, 1, O)

		return cache necessary for apply_recurrence
		"""
		carry = linen.LSTMCell.initialize_carry(key, (batch_size, ), embd_dim)
		init_obs = preprocess_observations_fn(init_obs, processor_params)
		init_obs_embds = init_obs_embder.apply(params['init_obs_embder_params'], init_obs)

		out_carry, _ = lstm.apply(params['lstm_params'], carry, init_obs_embds)
		return out_carry

	return SequenceModel(init=init, 
						apply_sequence=apply_sequence,
						apply_recurrence=apply_recurrence, 
						prime_recurrence=prime_recurrence)

def make_rnn_transition_network(
	obs_size: int,
	action_size: int,
	max_episode_length: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types
	.identity_observation_preprocessor,
	decoder_hidden_layer_sizes: Sequence[int] = (32, 32),
	embd_dim: int = 72,
	true_timesteps: bool = False,
	seed: int = 0) -> FeedForwardNetwork:

	rnn = SimpleRNN()

	# Embedders
	# init_obs_embder = Embed(embd_size=embd_dim, 
	# 						kernel_init=jax.nn.initializers.lecun_uniform())
	init_obs_embder = MLP(layer_sizes=list(decoder_hidden_layer_sizes) + [embd_dim],
						activation=linen.relu,
						kernel_init=jax.nn.initializers.lecun_uniform())

	# action_embder = Embed(embd_size=embd_dim,
	# 						kernel_init=jax.nn.initializers.lecun_uniform())
	action_embder = MLP(layer_sizes=list(decoder_hidden_layer_sizes) + [embd_dim],
						activation=linen.relu,
						kernel_init=jax.nn.initializers.lecun_uniform())

	# Decoder
	out_size = obs_size - 1 if true_timesteps else obs_size
	decoder = MLP(
		layer_sizes=list(decoder_hidden_layer_sizes) + [out_size],
		activation=linen.relu,
		kernel_init=jax.nn.initializers.lecun_uniform())

	def init(key):
		dummy_obs = jnp.zeros((1, obs_size))
		dummy_action = jnp.zeros((1, action_size))
		dummy_embd = jnp.zeros((1, embd_dim))
		dummy_sequence = jnp.zeros((1, 1, embd_dim))

		obs_key, action_key, timestep_key, decoder_key, rnn_key = jax.random.split(key, 5)

		init_carry = RNNCell.initialize_carry(rnn_key, (1,), embd_dim)
		rnn_params = rnn.init(rnn_key, init_carry, dummy_sequence)
		init_obs_embder_params = init_obs_embder.init(obs_key, dummy_obs)
		action_embder_params = action_embder.init(action_key, dummy_action)
		decoder_params = decoder.init(decoder_key, dummy_embd)

		transition_params = {'rnn_params': rnn_params, 'init_obs_embder_params': init_obs_embder_params,
							'action_embder_params': action_embder_params, 
							'decoder_params': decoder_params}
		return transition_params

	def apply_sequence(processor_params, params, init_obs, actions, key=None, train=True):
		"""
		init_obs of shape (B, 1, O)
		actions of shape	(B, L, A)

		returns observations of shape (B, L+1, O)
		"""
		batch_size = actions.shape[0]
		carry = RNNCell.initialize_carry(key, (batch_size, ), embd_dim)
		init_obs = preprocess_observations_fn(init_obs, processor_params)
		init_obs_embds = init_obs_embder.apply(params['init_obs_embder_params'], init_obs)

		action_embds = action_embder.apply(params['action_embder_params'], actions) # (B, L, H)
		sequence = jnp.concatenate((init_obs_embds, action_embds), axis=1)

		_, out_val = rnn.apply(params['rnn_params'], carry, sequence) # ((B, H), (B, H)), (B, L, H)
		obs_preds = decoder.apply(params['decoder_params'], out_val)

		if true_timesteps:
			sequence_length = actions.shape[1] + 1
			next_timesteps = jnp.repeat(jnp.arange(sequence_length)[None,...,None], repeats=batch_size, axis=0)
			init_times = init_obs[..., -1:]
			next_timesteps = next_timesteps + init_times
			obs_preds = jnp.concatenate((obs_preds, next_timesteps), axis=-1)
		return obs_preds

	def apply_recurrence(processor_params, params, actions, timesteps, cache, key=None, train=True):
		"""
		action of shape (B, 1, A)
		timesteps of shape (1,) - current timestep of the sequence
		cache is recurrent 

		returns next_obs (B, 1, O), and cache
		"""
		carry = cache
		action_embds = action_embder.apply(params['action_embder_params'], actions)
		out_carry, out_val = rnn.apply(params['rnn_params'], carry, action_embds) # ((B, H), (B, H)), (B, L, H)
		obs_preds = decoder.apply(params['decoder_params'], out_val)
		if true_timesteps:
			batch_size = actions.shape[0]
			next_timesteps = timesteps + 1.
			next_timesteps = jnp.repeat(next_timesteps[None, ..., None], repeats=batch_size, axis=0)
			obs_preds = jnp.concatenate((obs_preds, next_timesteps), axis=-1)
		return obs_preds, out_carry

	def prime_recurrence(processor_params, params, batch_size, unroll_length, init_obs, key, train=True):
		"""
		Primes the recurrence with a given initial observation
		init_obs of shape (B, 1, O)

		return cache necessary for apply_recurrence
		"""
		carry = RNNCell.initialize_carry(key, (batch_size, ), embd_dim)
		init_obs = preprocess_observations_fn(init_obs, processor_params)
		init_obs_embds = init_obs_embder.apply(params['init_obs_embder_params'], init_obs)

		out_carry, _ = rnn.apply(params['rnn_params'], carry, init_obs_embds)
		return out_carry

	return SequenceModel(init=init, 
						apply_sequence=apply_sequence,
						apply_recurrence=apply_recurrence, 
						prime_recurrence=prime_recurrence)

def make_s4_transition_network(
	obs_size: int,
	action_size: int,
	max_episode_length: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types
	.identity_observation_preprocessor,
	decoder_hidden_layer_sizes: Sequence[int] = (32, 32),
	n_layers: int = 3,
	embd_dim: int = 72,
	state_space_dim: int = 8,
	dropout: float = 0.1,
	true_timesteps: bool = False,
	seed: int = 0) -> FeedForwardNetwork:

	s4_args = {
		"N": state_space_dim,
		"l_max": max_episode_length
	}
	s4_model = BatchStackedModel(layer=S4Layer,
										layer_args=s4_args,
										d_output=embd_dim,
										d_model=embd_dim,
										n_layers=n_layers,
										dropout=dropout)

	recurrent_s4 = BatchStackedModel(layer=S4Layer,
										layer_args=s4_args,
										d_output=embd_dim,
										d_model=embd_dim,
										n_layers=n_layers,
										decode=True,
										training=False) #should this be set to True?
	recurrent_apply = partial(recurrent_s4.apply, mutable=['cache'])

	# Embedders
	init_obs_embder = Embed(embd_size=embd_dim, 
							kernel_init=jax.nn.initializers.lecun_uniform())
	action_embder = Embed(embd_size=embd_dim,
							kernel_init=jax.nn.initializers.lecun_uniform())

	# Decoder
	out_size = obs_size - 1 if true_timesteps else obs_size
	decoder = MLP(
		layer_sizes=list(decoder_hidden_layer_sizes) + [out_size],
		activation=linen.relu,
		kernel_init=jax.nn.initializers.lecun_uniform())

	def init(key):
		dummy_obs = jnp.zeros((1, obs_size))
		dummy_action = jnp.zeros((1, action_size))
		dummy_embd = jnp.zeros((1, embd_dim))
		dummy_sequence = jnp.zeros((1, max_episode_length, embd_dim))

		obs_key, action_key, s4_key, decoder_key, dropout_key = jax.random.split(key, 5)

		s4_params = s4_model.init({"params": s4_key, "dropout":	dropout_key}, dummy_sequence)
		init_obs_embder_params = init_obs_embder.init(obs_key, dummy_obs)
		action_embder_params = action_embder.init(action_key, dummy_action)
		decoder_params = decoder.init(decoder_key, dummy_embd)

		transition_params = {'s4_params': s4_params, 'init_obs_embder_params': init_obs_embder_params,
							'action_embder_params': action_embder_params,
							"decoder_params": decoder_params}
		return transition_params

	def apply_sequence(processor_params, params, init_obs, actions, key=None, train=True):
		"""
		init_obs of shape (B, 1, O)
		actions of shape	(B, L, A)

		returns observations of shape (B, L+1, O)
		"""
		dropout_key, key = jax.random.split(key)
		init_obs = preprocess_observations_fn(init_obs, processor_params)
		init_obs_embds = init_obs_embder.apply(params['init_obs_embder_params'], init_obs)

		action_embds = action_embder.apply(params['action_embder_params'], actions)
		sequence = jnp.concatenate((init_obs_embds, action_embds), axis=1)
		obs_preds = s4_model.apply(params['s4_params'], sequence, rngs={'dropout': dropout_key}) # (B, L+1, H)
		obs_preds = decoder.apply(params['decoder_params'], obs_preds)

		if true_timesteps:
			sequence_length = actions.shape[1] + 1
			batch_size = actions.shape[0]
			next_timesteps = jnp.repeat(jnp.arange(sequence_length)[None,...,None], repeats=batch_size, axis=0)
			init_times = init_obs[..., -1:]
			next_timesteps = next_timesteps + init_times
			obs_preds = jnp.concatenate((obs_preds, next_timesteps), axis=-1)

		return obs_preds


	def apply_recurrence(processor_params, params, actions, timesteps, cache, key=None, train=True):
		"""
		action of shape (B, 1, A)
		timesteps of shape (1,) - current timestep of the sequence
		cache is recurrent 

		returns next_obs (B, 1, O), and cache
		"""
		prime_vars, cache_vars = cache # prime_vars contains pre-computed discretized matrices. cache_vars are the previous hidden states

		action_embds = action_embder.apply(params['action_embder_params'], actions) # (B, 1, H)
		out, variables = recurrent_apply({'params': params['s4_params']['params'], "prime": prime_vars,
											"cache": cache_vars}, action_embds)

		cache_vars = variables['cache']
		new_cache = (prime_vars, cache_vars)
		obs_preds = decoder.apply(params['decoder_params'], out)

		if true_timesteps:
			batch_size = actions.shape[0]
			next_timesteps = timesteps + 1.
			next_timesteps = jnp.repeat(next_timesteps[None, ..., None], repeats=batch_size, axis=0)
			obs_preds = jnp.concatenate((obs_preds, next_timesteps), axis=-1)
			
		return obs_preds, new_cache

	def prime_recurrence(processor_params, params, batch_size, unroll_length, init_obs, key, train=True):
		"""
		Primes the recurrence with a given initial observation
		init_obs of shape (B, 1, O)

		return cache necessary for apply_recurrence
		"""
		init_key, key = jax.random.split(key)
		init_x = jnp.zeros((batch_size, unroll_length, embd_dim))
		variables = recurrent_s4.init(init_key, init_x)
		variables = {
			"params": params['s4_params']['params'],
			"cache": variables['cache'].unfreeze(),
			'prime': variables['prime'].unfreeze()
		}

		_, prime_vars = recurrent_s4.apply(variables, init_x, mutable=['prime'])
		prime_vars = prime_vars["prime"]
		cache_vars = variables["cache"]
		
		init_obs_embds = init_obs_embder.apply(params['init_obs_embder_params'], init_obs)
		_, variables = recurrent_apply({'params': params['s4_params']['params'], "prime": prime_vars,
						"cache": cache_vars}, init_obs_embds)
		primed_cache = (prime_vars, variables['cache'])

		return primed_cache

	return SequenceModel(init=init, 
						apply_sequence=apply_sequence,
						apply_recurrence=apply_recurrence, 
						prime_recurrence=prime_recurrence)

def make_reward_network(
	obs_size: int,
	action_size: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types
	.identity_observation_preprocessor,
	hidden_layer_sizes: Sequence[int] = (256, 256),
	activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
	"""Creates reward network"""
	reward_module = MLP(
		layer_sizes=list(hidden_layer_sizes) + [1],
		activation=activation,
		kernel_init=jax.nn.initializers.lecun_uniform())

	def apply(processor_params, reward_params, obs, actions):
		obs = preprocess_observations_fn(obs, processor_params)
		return reward_module.apply(reward_params, jnp.concatenate((obs, actions), axis=-1))

	dummy_obs = jnp.zeros((1, obs_size))
	dummy_action = jnp.zeros((1, action_size))
	return FeedForwardNetwork(
		init=lambda key: reward_module.init(key, jnp.concatenate((dummy_obs, dummy_action), axis=-1)), apply=apply)

def make_critic_network(
	obs_size: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types
	.identity_observation_preprocessor,
	hidden_layer_sizes: Sequence[int] = (128, 128),
	activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
	"""Creates Value function network"""

	critic_module = MLP(
		layer_sizes=list(hidden_layer_sizes)+ [1],
		activation=activation,
		kernel_init=jax.nn.initializers.lecun_uniform())

	def apply(processor_params, critic_params, obs):
		obs = preprocess_observations_fn(obs, processor_params)
		return critic_module.apply(critic_params, obs)

	dummy_obs = jnp.zeros((1, obs_size))
	return FeedForwardNetwork(
		init=lambda key: critic_module.init(key, dummy_obs), apply=apply)



