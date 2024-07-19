from typing import Sequence, Tuple, Callable, Any

import flax
from flax import linen

from brax.training import distribution
from brax.training import types

from src.brax import networks

@flax.struct.dataclass
class ARMNetworks:
	policy_network: networks.FeedForwardNetwork
	transition_network: networks.FeedForwardNetwork
	reward_network: networks.FeedForwardNetwork
	parametric_action_distribution: distribution.ParametricDistribution
	critic_network: networks.FeedForwardNetwork
	
def make_inference_fn(arm_networks: ARMNetworks):
  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False,
                  get_dist: bool  =  False) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample) -> Tuple[types.Action, types.Extra]:
      logits = arm_networks.policy_network.apply(*params, observations)
      if deterministic:
        mode = arm_networks.parametric_action_distribution.create_dist(logits).loc
        return arm_networks.parametric_action_distribution.postprocess(mode), {'entropy': arm_networks.parametric_action_distribution.entropy(logits, key_sample)}

      if not get_dist:
        return arm_networks.parametric_action_distribution.sample(
          logits, key_sample), {'entropy': arm_networks.parametric_action_distribution.entropy(logits, key_sample)}
      else:
        dist = arm_networks.parametric_action_distribution.create_dist(logits)
        return arm_networks.parametric_action_distribution.sample(
          logits, key_sample), {'entropy': arm_networks.parametric_action_distribution.entropy(logits, key_sample),
                            'loc': dist.loc,
                            'scale': dist.scale}


    return policy

  return make_policy

def make_arm_networks(
	observation_size: int,
	action_size: int,
	max_episode_length: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
	hidden_layer_sizes: Sequence[int] = (64,) * 2,
	decoder_hidden_layer_sizes: Sequence[int] = (32, 32),
	embd_dim: int = 72,
	input_observations=False,
	transformer_nlayers: int = 3,
	transformer_nheads: int = 3,
	transformer_pdrop: float = 0.1,
	seed: int = 0,
	true_timesteps: bool = True,
	activation: networks.ActivationFn = linen.relu) -> ARMNetworks:

	parametric_action_distribution = distribution.NormalTanhDistribution(
		event_size=action_size) ## VERIFY: This bounds actions between [-1, 1]? 
	policy_network = networks.make_policy_network(
		parametric_action_distribution.param_size,
		observation_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)
	reward_network = networks.make_reward_network(
		observation_size,
		action_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)
	critic_network = networks.make_critic_network(
		observation_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)

	transition_network = networks.make_gpt_transition_network(
		observation_size,
		observation_size + action_size if input_observations else action_size,
		max_episode_length,
		preprocess_observations_fn=preprocess_observations_fn,
		decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
		embd_dim=embd_dim,
		transformer_nlayers=transformer_nlayers,
		transformer_nheads=transformer_nheads,
		transformer_pdrop=transformer_pdrop,
		true_timesteps=true_timesteps,
		seed=seed)

	return ARMNetworks(
		policy_network=policy_network,
		transition_network=transition_network,
		reward_network=reward_network,
		critic_network=critic_network,
		parametric_action_distribution=parametric_action_distribution)

def make_arm_lstm_networks(
	observation_size: int,
	action_size: int,
	max_episode_length: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
	hidden_layer_sizes: Sequence[int] = (64,) * 2,
	decoder_hidden_layer_sizes: Sequence[int] = (32, 32),
	embd_dim: int = 72,
	input_observations=False,
	seed: int = 0,
	true_timesteps: bool=True,
	activation: networks.ActivationFn = linen.relu) -> ARMNetworks:

	parametric_action_distribution = distribution.NormalTanhDistribution(
		event_size=action_size) ## VERIFY: This bounds actions between [-1, 1]? 
	policy_network = networks.make_policy_network(
		parametric_action_distribution.param_size,
		observation_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)
	reward_network = networks.make_reward_network(
		observation_size,
		action_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)
	critic_network = networks.make_critic_network(
		observation_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)

	transition_network = networks.make_lstm_transition_network(
		observation_size,
		observation_size + action_size if input_observations else action_size,
		max_episode_length,
		preprocess_observations_fn=preprocess_observations_fn,
		decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
		embd_dim=embd_dim,
		true_timesteps=true_timesteps,
		seed=seed)

	return ARMNetworks(
		policy_network=policy_network,
		transition_network=transition_network,
		reward_network=reward_network,
		critic_network=critic_network,
		parametric_action_distribution=parametric_action_distribution)

def make_arm_rnn_networks(
	observation_size: int,
	action_size: int,
	max_episode_length: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
	hidden_layer_sizes: Sequence[int] = (64,) * 2,
	decoder_hidden_layer_sizes: Sequence[int] = (32, 32),
	embd_dim: int = 72,
	input_observations=False,
	seed: int = 0,
	true_timesteps: bool = True,
	activation: networks.ActivationFn = linen.relu) -> ARMNetworks:

	parametric_action_distribution = distribution.NormalTanhDistribution(
		event_size=action_size) ## VERIFY: This bounds actions between [-1, 1]? 
	policy_network = networks.make_policy_network(
		parametric_action_distribution.param_size,
		observation_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)
	reward_network = networks.make_reward_network(
		observation_size,
		action_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)
	critic_network = networks.make_critic_network(
		observation_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)

	transition_network = networks.make_rnn_transition_network(
		observation_size,
		observation_size + action_size if input_observations else action_size,
		max_episode_length,
		preprocess_observations_fn=preprocess_observations_fn,
		decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
		embd_dim=embd_dim,
		true_timesteps=true_timesteps,
		seed=seed)

	return ARMNetworks(
		policy_network=policy_network,
		transition_network=transition_network,
		reward_network=reward_network,
		critic_network=critic_network,
		parametric_action_distribution=parametric_action_distribution)

def make_arm_s4_networks(
	observation_size: int,
	action_size: int,
	max_episode_length: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
	hidden_layer_sizes: Sequence[int] = (64,) * 2,
	decoder_hidden_layer_sizes: Sequence[int] = (32, 32),
	embd_dim: int = 72,
	input_observations=False,
	n_layers: int = 3,
	state_space_dim: int = 8,
	dropout: float = 0.1,
	true_timesteps=True,
	seed: int = 0,
	activation: networks.ActivationFn = linen.relu) -> ARMNetworks:

	parametric_action_distribution = distribution.NormalTanhDistribution(
		event_size=action_size) ## VERIFY: This bounds actions between [-1, 1]? 
	policy_network = networks.make_policy_network(
		parametric_action_distribution.param_size,
		observation_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)
	reward_network = networks.make_reward_network(
		observation_size,
		action_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)
	critic_network = networks.make_critic_network(
		observation_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)

	transition_network = networks.make_s4_transition_network(
		observation_size,
		observation_size + action_size if input_observations else action_size,
		max_episode_length,
		preprocess_observations_fn=preprocess_observations_fn,
		embd_dim=embd_dim,
		decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
		n_layers=n_layers,
		state_space_dim=state_space_dim,
		dropout=dropout,
		seed=seed)

	return ARMNetworks(
		policy_network=policy_network,
		transition_network=transition_network,
		reward_network=reward_network,
		critic_network=critic_network,
		parametric_action_distribution=parametric_action_distribution)