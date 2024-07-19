import jax.numpy as jnp
import jax
import tqdm
import numpy as np
import random

import numpy as np


def get_output(training_state, transition_network, observations, actions, key):
    init_obs = observations[:, 0:1, :]
    next_obs_predictions = transition_network.apply_sequence(training_state.preprocessor_params,
                                                            training_state.transition_params,
                                                            init_obs, actions, 
                                                             key=key, train=False)[:, 1:, :] # exclude init obs pred
    return next_obs_predictions

def get_pred_errors(training_state, transition_network, dataset, sample_size, key):
    sampled_episodes = dataset.random_episodes(sample_size)
    obs = jnp.array(sampled_episodes['obs'])
    next_obs = jnp.array(sampled_episodes['obs2'])
    actions = jnp.array(sampled_episodes['act'])
    skey, key = jax.random.split(key)
    obs_preds = get_output(training_state, transition_network, obs, actions, skey)
    
    mse = np.mean((next_obs - obs_preds)**2, axis=-1)
    per_timestep_errors = np.mean(mse, axis=0)
    
    return per_timestep_errors, np.mean(mse)


def get_one_grads(actions, observations, training_state, transition_network, key):
    def get_one_output(first_action, other_actions, observations, training_state, transition_network, key):
        # for just one trajectory
        # gradient with respect to first action only
        # action input as shape (L), but need to resize to (B, L, 1)
        init_obs = observations[:, 0:1, :]
        actions = jnp.append(first_action, other_actions)
        actions = jnp.expand_dims(actions, axis=(0, -1))
        next_obs_predictions = transition_network.apply_sequence(training_state.preprocessor_params,
                                                                training_state.transition_params,
                                                                init_obs, actions, 
                                                                 key=key, train=False)[:, 1:, :] # exclude init obs pred
        return next_obs_predictions[0]

    
    J = jax.jacfwd(get_one_output)(actions[0], actions[1:], observations, training_state, transition_network, key)
    return J

def get_batch_grads(training_state, transition_network, dataset, num_samples, key):
    sampled_episodes = dataset.random_episodes(num_samples)
    obs = jnp.array(sampled_episodes['obs'])
    actions = jnp.array(sampled_episodes['act'])
    skey, key = jax.random.split(key)
    Js = jax.vmap(get_one_grads, in_axes=(0, 0, None, None, None), out_axes=0)(jnp.squeeze(actions),
                                                                          jnp.expand_dims(obs, axis=1),
                                                                         training_state,
                                                                          transition_network,
                                                                          key)
    return Js

def get_first_actiongrad_stats(Js):
    # for the toy-ca problem
    length = Js.shape[1]
    true_js = np.zeros((length, 3))
    true_js[:, 0] = 1
    mean_Js = np.mean(Js, axis=0)
    
    # print(Js.shape, true_js.shape)
    cosims = []
    
    l2s = np.sum((mean_Js - true_js)**2, axis=1)
    deviations = np.std(Js, axis=0)
    deviations = np.mean(deviations, axis=1)
    for i in range(length):
        cosim = np.dot(mean_Js[i, :], true_js[i, :])/(np.linalg.norm(mean_Js[i, :]) * np.linalg.norm(true_js[i, :]))
        cosims.append(cosim)
        
    return cosims, l2s, deviations