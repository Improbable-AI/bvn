import os
import random

import gym
import numpy as np
import torch
from mpi4py import MPI
from ddpg_her.ddpg_agent import ddpg_agent

from ddpg_her import Args, MetricArgs

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              }
    params['max_timesteps'] = env._max_episode_steps
    return params


def main(dep: Args = {}):
    from ml_logger import logger, RUN
    
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    Args._update(dep)
    MetricArgs._update(dep)

    os.environ["ML_LOGGER_ROOT"] = f"{os.getcwd()}/results/{Args.agent_type}/{Args.critic_type}/{Args.env_name}/{Args.seed}"
    print("Dep", dep)

    if not RUN.resume:
        logger.remove("metrics.pkl", 'videos', 'models', 'outputs.log')

    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.log_params(Args=vars(Args), MetricArgs=vars(MetricArgs))
        logger.log_text("""
                keys:
                - run.status
                - Args.env_name
                - host.hostname
                charts:
                - yKeys: [test/success]
                  xKey: env_steps
                  yDomain: [0, 1]
                - yKeys: [test/success]
                  xKey: epoch
                  yDomain: [0, 1]
                 """, ".charts.yml", dedent=True, overwrite=True)

    # create the ddpg_agent
    env = gym.make(Args.env_name)
    # set random seeds for reproduce
    env.seed(Args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(Args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(Args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(Args.seed + MPI.COMM_WORLD.Get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment
    ddpg_trainer = {'ddpg': ddpg_agent}[Args.agent_type](Args, env, env_params, MetricArgs)
    ddpg_trainer.learn()

if __name__ == "__main__":
    main()