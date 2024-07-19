import sys, os, time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

t0 = time.time()
pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    jobid = str(os.environ["SLURM_JOB_ID"])
else:
    jobid = pid
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(0.7) # instead of default 0.9 see:  https://github.com/google/jax/issues/13504
from brax import envs
from absl import flags, app
from ml_collections import config_flags
from ml_collections import ConfigDict
import pickle


import src.brax.svginf.train as svg
import src.brax.arm.train as arm
import src.brax.sac.train as sac
from src.misc.helper_methods import moving_avg

from src.brax.custom_envs.myriad.lenhart.cancer_treatment import CancerTreatment
from src.brax.custom_envs.myriad.lenhart.bear_populations import BearPopulations
from src.brax.custom_envs.myriad.lenhart.bioreactor import Bioreactor
from src.brax.custom_envs.myriad.lenhart.hiv_treatment import HIVTreatment
from src.brax.custom_envs.myriad.lenhart.mould_fungicide import MouldFungicide


from src.brax.custom_envs.myriad.lenhart.bacteria import Bacteria # Terminal cost
from src.brax.custom_envs.myriad.lenhart.glucose import Glucose #?
from src.brax.custom_envs.myriad.lenhart.harvest import Harvest # requires timestep
from src.brax.custom_envs.myriad.lenhart.invasive_plant import InvasivePlant # discrete time, not compatible
from src.brax.custom_envs.myriad.lenhart.predator_prey import PredatorPrey # Terminal cost
from src.brax.custom_envs.myriad.lenhart.timber_harvest import TimberHarvest # requires timestep

from src.brax.custom_envs.myriad.brax_wrapper import MyriadEnv

from src.utils import logger, system

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config_env',  './configs/envs/exp4_toy_ca.py', 'file path to the environment configuration.', lock_config=False)
config_flags.DEFINE_config_file('config_rl', './configs/rl/arm_default.py', 'file path to the RL algorithm configuration.', lock_config=False)

flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_integer('seed', 42, 'seed int')

name_to_class = {
    'CANCER': CancerTreatment,
    'BEARPOPULATIONS': BearPopulations,
    'BIOREACTOR': Bioreactor,
    'HIVTREATMENT': HIVTreatment,
    'MOULDFUNGICIDE': MouldFungicide,
    'BACTERIA': Bacteria,
    'PREDATORPREY': PredatorPrey,
    'GLUCOSE': Glucose,
    'HARVEST': Harvest,
    'TIMBERHARVEST': TimberHarvest,
    'INVASIVEPLANT': InvasivePlant
}

def remove_key(cfg, k):
    dct = cfg.to_dict()
    dct.pop(k)
    return ConfigDict(dct)

def progress_fn(step: int, metrics):
    logger.record_step('env_steps', step)
    for k, v in metrics.items():
        logger.record_tabular(k, v)
    logger.dump_tabular()

def main(argv):
    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    
    
    print('saving files')
    
    # logger
    env_name = "{0}-{1}".format(config_env.env_type, config_env.length)
    run_name = f"{env_name}/"
    run_name += f"{system.now_str()}+{jobid}-{pid}"
    format_strs = ['csv']
    if FLAGS.debug:
        format_strs.extend(['stdout', 'logs'])
    log_path = os.path.join('./logs/myriad-icml-distractors', run_name)
    logger.configure(dir=log_path, format_strs=format_strs)
    # write flags to a txt
    key_flags = FLAGS.get_key_flags_for_module(argv[0])
    with open(os.path.join(log_path, "flags.txt"), "w") as text_file:
        text_file.write("\n".join(f.serialize() for f in key_flags) + "\n")
    # write flags to pkl
    with open(os.path.join(log_path, "flags.pkl"), "wb") as f:
        pickle.dump(FLAGS.flag_values_dict(), f)
        
    print('creating env')
    
    ## creating env
    base_env = name_to_class[config_env.env_type]()
    if config_env.constant_dt:
        default_dt = base_env.T / 100.
        env = MyriadEnv(base_env, config_env.length, default_dt, config_env.distractor_dims)
        eval_env = MyriadEnv(base_env, config_env.length, default_dt, config_env.distractor_dims)

    else:
        default_time = base_env.T
        dt = default_time/config_env.length
        env = MyriadEnv(base_env, config_env.length, dt, config_env.distractor_dims)
        eval_env = MyriadEnv(base_env, config_env.length, dt, config_env.distractor_dims)


    if config_rl.alg == 'arm':
        alg = arm
        # max unroll, no critics
        if config_rl.unroll_length == -1:
            config_rl.unroll_length = config_env.length
        if config_rl.sequence_model_name == 'gpt':
            config_rl.sequence_model_params = {'name': config_rl.sequence_model_name,
                                                "transformer_nlayers" : 2,
                                                "transformer_nheads": 3,
                                                "transformer_pdrop": 0.1}
            # config_rl.sequence_model_params = {'name': config_rl.sequence_model_name,
            #                                     "transformer_nlayers" : config_rl.transformer_nlayers,
            #                                     "transformer_nheads": config_rl.transformer_nheads,
            #                                     "transformer_pdrop": config_rl.transformer_pdrop}
            # config_rl = remove_key(config_rl, 'transformer_nlayers')
            # config_rl = remove_key(config_rl, 'transformer_nheads')
            # config_rl = remove_key(config_rl, 'transformer_pdrop')
        else:
            config_rl.sequence_model_params = {'name': config_rl.sequence_model_name}
        
        config_rl = remove_key(config_rl, 'sequence_model_name')
        
            
    elif config_rl.alg == 'svg':
        alg = svg
        # Make sure imagined trajectories dont get to terminal states
        config_rl.chunk_length = config_rl.unroll_length + 1

        # full unroll, no critic
        if config_rl.unroll_length == -1:
            config_rl.unroll_length = config_env.length
            config_rl.chunk_length = config_env.length
            config_rl.bootstrap = 0.

    elif config_rl.alg == 'sac':
        alg = sac
    
    # for logging purposes, it only works when evaluation steps coincide with update steps
    if config_rl.eval_every == -1:
        config_rl.eval_every = max(config_rl.dynamics_update_every, config_rl.policy_update_every) * 2
    
    # removing unnecessary key
    config_rl = remove_key(config_rl, 'alg')
    
    config_rl.seed = int(pid)

    _ = alg.train(env, eval_env, config_env.length,
               progress_fn=progress_fn,
               **config_rl)

if __name__ == '__main__':
    app.run(main)
