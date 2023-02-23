import random
from functools import partial

import gym
import numpy as np
import torch

from rl import Args
from rl.agent import Agent
from rl.algo.core import Algo
from rl.learner import Learner
from rl.replays.her import Replay
from rl.utils import vec_env

from gym.wrappers import TimeLimit
from gym import Wrapper

from collections.abc import Mapping
from typing import Any


def get_env_params(env_id):
    env = gym.make(env_id)
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    reward_func = env.compute_reward
    del env  # to avoid memory leak
    return params, reward_func


def get_env_with_id(num_envs, env_id):
    vec_fn = vec_env.DummyVecEnv if Args.debug else vec_env.SubprocVecEnv
    env = vec_fn([lambda: gym.make(env_id) for _ in range(num_envs)])
    env.name = env_id
    return env


def get_env_with_fn(num_envs, env_fn, *args, **kwargs):
    vec_fn = vec_env.DummyVecEnv if Args.debug else vec_env.SubprocVecEnv
    return vec_fn([lambda: env_fn(*args, **kwargs) for _ in range(num_envs)])


def launch(deps=None):
    Args._update(deps)

    torch.set_num_threads(2)

    # rank = mpi_utils.get_rank()
    # seed = Args.seed + rank * Args.n_workers
    random.seed(Args.seed)
    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    if Args.cuda:
        torch.cuda.manual_seed(Args.seed)

    env_params, reward_func = get_env_params(Args.env_name)

    env = get_env_with_id(num_envs=Args.n_workers, env_id=Args.env_name)
    env.seed(Args.seed)

    if Args.test_env_name is None:
        test_env = None
    else:
        test_env = get_env_with_id(num_envs=Args.n_workers, env_id=Args.test_env_name)
        test_env.seed(Args.seed + 100)

    agent = Agent(env_params, Args)
    replay = Replay(env_params, Args, reward_func)
    learner = Learner(agent, Args)

    algo = Algo(env=env, test_env=test_env, env_params=env_params, args=Args, agent=agent, replay=replay,
            learner=learner, reward_func=reward_func)

    return algo
