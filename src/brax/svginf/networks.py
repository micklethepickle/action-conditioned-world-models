from typing import Sequence, Tuple

import flax
from flax import linen

from brax.training import distribution
from brax.training import types

from src.brax import networks

@flax.struct.dataclass
class SVGNetworks:
	policy_network: networks.FeedForwardNetwork
	transition_network: networks.FeedForwardNetwork
	reward_network: networks.FeedForwardNetwork
	critic_network: networks.FeedForwardNetwork
	parametric_action_distribution: distribution.ParametricDistribution
	
def make_inference_fn(svg_networks: SVGNetworks):
  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample) -> Tuple[types.Action, types.Extra]:
      logits = svg_networks.policy_network.apply(*params, observations)
      if deterministic:
      	mode = svg_networks.parametric_action_distribution.create_dist(logits).loc
      	return svg_networks.parametric_action_distribution.postprocess(mode), {'entropy': svg_networks.parametric_action_distribution.entropy(logits, key_sample)}
      return svg_networks.parametric_action_distribution.sample(
          logits, key_sample), {'entropy': svg_networks.parametric_action_distribution.entropy(logits, key_sample)}

    return policy

  return make_policy

def make_svg_networks(
	observation_size: int,
	action_size: int,
	preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
	hidden_layer_sizes: Sequence[int] = (64,) * 2,
	activation: networks.ActivationFn = linen.relu,
	true_timesteps: bool = True,
	difference_transition: bool = True) -> SVGNetworks:

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
	transition_network = networks.make_transition_network(
		observation_size,
		action_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation,
		true_timesteps=true_timesteps,
		difference_transition=difference_transition)

	critic_network = networks.make_critic_network(
		observation_size,
		preprocess_observations_fn=preprocess_observations_fn,
		hidden_layer_sizes=hidden_layer_sizes,
		activation=activation)

	return SVGNetworks(
		policy_network=policy_network,
		transition_network=transition_network,
		reward_network=reward_network,
		critic_network=critic_network,
		parametric_action_distribution=parametric_action_distribution,)