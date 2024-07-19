import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import math

def moving_avg(stuff, window):
    return np.convolve(stuff, np.ones(window)/window, mode='valid')

def grad_norm(grads):
    leaves, _ = jax.tree_util.tree_flatten(grads)
    return jnp.sqrt(sum([jnp.sum(leaf ** 2) for leaf in leaves]))

def clip_grad_norm(grad, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad)))
    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(clip, grad)

def mse(predictions, true):
    return 0.5*jnp.mean(jnp.square(true - predictions)) 

def detach(x):
    return jax.lax.stop_gradient(x)

def plot_many(experiments, xs=None, ax=None, label=None, color=None):
    if ax is None:
        ax = plt
    mean_exp = jnp.mean(experiments, axis=0)
    std_exp = jnp.std(experiments, axis=0)
    if xs is None:
        xs = range(len(mean_exp))
    ax.plot(mean_exp, color=color, label=label)
    ax.fill_between(range(len(experiments[0])), mean_exp + std_exp, mean_exp - std_exp, color=color, alpha=0.1)

def make_plots(plots, subplot_size, labels, ncols=3, consistent_coloring=False):
    """
    plots is 3d (or 2d) array. First dim is figure index, second dim is list of plots
    labels is 2d (or 1d) array. First dim is figure, second dim are the labels
    subplot_size: tuple size of each subplot
    """
    plots = np.array(plots)
    colors = ['#E26D5C', '#7E78D2', '#FFD166', '#8FC0A9', '#690375']
    num_subplots = len(plots)
    nrows = math.ceil(num_subplots / ncols)
    ncols = ncols if num_subplots >= ncols else num_subplots
    figure, axis = plt.subplots(nrows, ncols, figsize=(subplot_size[0]*ncols, subplot_size[1]*nrows))
    if nrows == 1 and ncols == 1:
        axis = [axis]
    else:
        axis = axis.flatten()
    
    num_plots = 0
    for i, p in enumerate(plots):
        if len(p.shape) > 1:
            for j,sp in enumerate(p):
                c = colors[j] if consistent_coloring else colors[num_plots % len(colors)]
                axis[i].plot(sp, color=c, label=labels[i][j])
                num_plots += 1
        else:
            axis[i].plot(p, color=colors[num_plots % len(colors)], label=labels[i])
            num_plots += 1
        axis[i].grid()
        axis[i].legend()

    return figure, axis

def target_update(new_params, target_params, tau: float):
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), new_params, target_params
    )
    return new_target_params

def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = 1/S1
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = jnp.sum(iS1 * S0)
    det_term  = jnp.sum(jnp.log(S1)) - jnp.sum(jnp.log(S0))
    quad_term = jnp.sum( (diff*diff) * iS1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 




import itertools
def construct_jobs(grid):
    jobs = []
    individual_options = [
        [{key: value} for value in values] for key, values in grid.items()
    ]
    product_options = list(itertools.product(*individual_options))
    jobs += [
        {k: v for d in option_set for k, v in d.items()}
        for option_set in product_options
    ]
    return jobs


def find_best(df, sweeped_hparams, num_seeds=10):
    """
    sweeped_hparams {k: v} where k are the hparams and v is a list of values to check
    """
    num_final_evals = 1
    
    hparam_configs = construct_jobs(sweeped_hparams)
    
    
    best_config = None
    best_mean = -np.inf
    best_returns = None
    
    for config in hparam_configs:
        filtered_df = df
        for k in config:
            value = config[k]
            filtered_df = filtered_df[filtered_df[k] == value]

        if len(np.array(filtered_df['env_steps'])) < 5: # 5 is arbitary low number
            print('MISSING {0} COMPLETLY')
        else:
            final_step = np.array(filtered_df['env_steps'])[-num_final_evals]
        
            filtered_df = df
            for k in config:
                value = config[k]
                filtered_df = filtered_df[filtered_df[k] == value]
            rewards = filtered_df[filtered_df['env_steps'] >= final_step]['eval/episode_reward']
            end_performances = np.array(rewards)
            
            if len(end_performances) < (num_final_evals * num_seeds):
                print('missing {0} results'.format(num_seeds - len(end_performances)))

            if np.mean(end_performances) > best_mean:
                best_returns = end_performances
                best_mean = np.mean(end_performances)
                best_config = config
        
            
    return best_config, best_mean, best_returns



