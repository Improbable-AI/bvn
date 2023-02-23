import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rl.normalizer import Normalizer
from rl.utils import net_utils

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        input_dim = env_params['obs'] + env_params['goal']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids,
            activation=args.activ, output_activation=args.activ)
        self.mean = nn.Linear(args.hid_size, env_params['action'])

    def forward(self, inputs):
        outputs = self.net(inputs)
        mean = self.mean(outputs)
        pi_action = torch.tanh(mean) * self.act_limit
        return pi_action


class Critic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids + [1],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        q_inputs = torch.cat([pi_inputs, actions / self.act_limit], dim=-1)
        q_values = self.net(q_inputs).squeeze()
        return q_values

class StateAsymMetricCritic(nn.Module):
    '''
    Instead of phi(f(s,a), g) we use phi(s, g) to promote full-rank.
    '''
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        embed_dim = args.metric_embed_dim
        self.f = net_utils.mlp(
            [env_params['obs'] + env_params['action']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        self.phi = net_utils.mlp(
            [env_params['obs'] + env_params['goal']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        # f(s, a)
        f_embeds = self.f_embed(obses, actions)

        # \varphi(f(s,a), g)
        phi_embeds = self.state_phi_embed(obses, goals)
        # phi_embeds = self.phi_embed(f_embeds, goals)

        # ||f(s, a) - \varphi(f(s, a), g)||
        embed_dist = torch.linalg.norm(f_embeds - phi_embeds, dim=-1)

        q_values = (-embed_dist).squeeze()

        return q_values

    def f_embed(self, obses, actions):
        f_inputs = torch.cat([obses, actions / self.act_limit], dim=-1)
        f_embeds = self.f(f_inputs)
        return f_embeds

    def state_phi_embed(self, obses, goals):
        phi_inputs = torch.cat([obses, goals], dim=-1)
        phi_embeds = self.phi(phi_inputs)
        return phi_embeds

class AsymMetricCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        # input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        self.state_action_encoder_net = net_utils.mlp(
            [env_params['obs'] + env_params['action']] + [args.hid_size] * args.n_hids + [16],
            activation=args.activ)
        self.goal_encoder_net = net_utils.mlp(
            [env_params['goal']] + [32] * args.n_hids + [16],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        # f(s, a)
        f_embeds = self.f_embed(obses, actions)

        # \varphi(f(s,a), g)
        phi_embeds = self.phi_embed(f_embeds.detach(), goals)
        # phi_embeds = self.phi_embed(f_embeds, goals)

        # ||f(s, a) - \varphi(f(s, a), g)||
        embed_dist = torch.linalg.norm(f_embeds - phi_embeds, dim=-1)

        q_values = (-embed_dist).squeeze()

        return q_values

    def f_embed(self, obses, actions):
        f_inputs = torch.cat([obses, actions / self.act_limit], dim=-1)
        f_embeds = self.f(f_inputs)
        return f_embeds

    def phi_embed(self, f_embeds, goals):
        phi_inputs = torch.cat([f_embeds, goals], dim=-1)
        phi_embeds = self.phi(phi_inputs)
        return phi_embeds

class SymMetricCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        embed_dim = args.metric_embed_dim
        self.f = net_utils.mlp(
            [env_params['obs'] + env_params['action']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        self.phi = net_utils.mlp(
            [env_params['goal']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        state_action_encoder_inputs = torch.cat([obses, actions / self.act_limit], dim=-1)

        state_action_embeds = self.state_action_encoder_net(state_action_encoder_inputs)
        if ret == 'sa_embed':
            return state_action_embeds

        goal_embeds = self.goal_encoder_net(goals)
        if ret == 'g_embed':
            return goal_embeds

        if ret == 'embed':
            return state_action_embeds, goal_embeds
        else:
            state_action_to_goal_distance = torch.linalg.norm(state_action_embeds - goal_embeds, dim=-1)
            q_values = (-state_action_to_goal_distance).squeeze()
            if ret == 'value_and_embed':
                return q_values, state_action_embeds, goal_embeds
            else:
                return q_values


class FSAGMetricCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        embed_dim = args.metric_embed_dim
        self.f = net_utils.mlp(
            [env_params['obs'] + env_params['action'] + env_params['goal']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        self.phi = net_utils.mlp(
            [env_params['goal']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        # f(s, a, g)
        f_embeds = self.fsag_f_embed(obses, actions, goals)

        # \varphi(g)
        phi_embeds = self.fsag_phi_embed(goals)

        # ||f(s, a, g) - \varphi(g)||
        embed_dist = torch.linalg.norm(f_embeds - phi_embeds, dim=-1)

        q_values = (-embed_dist).squeeze()

        return q_values

    def fsag_f_embed(self, obses, actions, goals):
        f_inputs = torch.cat([obses, actions / self.act_limit, goals], dim=-1)
        f_embeds = self.f(f_inputs)
        return f_embeds

    def fsag_phi_embed(self, goals):
        phi_embeds = self.phi(goals)
        return phi_embeds

class BaseAgent:
    def __init__(self, env_params, args, name='agent'):
        self.env_params = env_params
        self.args = args
        self._save_file = str(name) + '.pt'

    @staticmethod
    def to_2d(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def to_tensor(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if self.args.cuda:
            x = x.cuda()
        return x

    @property
    def device(self):
        return torch.device("cuda" if self.args.cuda else "cpu")

    def get_actions(self, obs, goal):
        raise NotImplementedError

    def get_pis(self, obs, goal):
        raise NotImplementedError

    def get_qs(self, obs, goal, actions, **kwargs):
        raise NotImplementedError

    def forward(self, obs, goal, *args, **kwargs):
        """ return q_pi, pi """
        raise NotImplementedError

    def target_update(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def save(self, path):
        if mpi_utils.is_root():
            state_dict = self.state_dict()
            save_path = osp.join(path, self._save_file)
            torch.save(state_dict, save_path)

    def load(self, path):
        load_path = osp.join(path, self._save_file)
        state_dict = torch.load(load_path)
        self.load_state_dict(state_dict)


class Agent(BaseAgent):
    def __init__(self, env_params, args, name='agent'):
        super().__init__(env_params, args, name=name)

        CriticCls = {'td': Critic,
                    'asym_metric': AsymMetricCritic,
                    'sym_metric': SymMetricCritic,
                    'fsag_metric': FSAGMetricCritic,
                    'state_asym_metric': StateAsymMetricCritic}[args.critic_type]

        self.actor = Actor(env_params, args)
        self.critic = CriticCls(env_params, args)

        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_networks(self.actor)
        #     mpi_utils.sync_networks(self.critic)

        self.actor_targ = Actor(env_params, args)
        self.critic_targ = CriticCls(env_params, args)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

        net_utils.set_requires_grad(self.actor_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic_targ, allow_grad=False)

        if self.args.cuda:
            self.cuda()

        self.o_normalizer = Normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_normalizer = Normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()
        self.actor_targ.cuda()
        self.critic_targ.cuda()

    def _preprocess_inputs(self, obs, goal):
        # add conditional here
        obs = self.to_2d(obs)
        goal = self.to_2d(goal)
        if self.args.clip_inputs:
            obs = np.clip(obs, -self.args.clip_obs, self.args.clip_obs)
            goal = np.clip(goal, -self.args.clip_obs, self.args.clip_obs)
        return obs, goal

    def _process_inputs(self, obs, goal):
        if self.args.normalize_inputs:
            obs = self.o_normalizer.normalize(obs)
            goal = self.g_normalizer.normalize(goal)
        return self.to_tensor(np.concatenate([obs, goal], axis=-1))

    def get_actions(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        with torch.no_grad():
            actions = self.actor(inputs).cpu().numpy().squeeze()
        return actions

    def get_pis(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        pis = self.actor(inputs)
        return pis

    def get_qs(self, obs, goal, actions, **kwargs):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        actions = self.to_tensor(actions)
        return self.critic(inputs, actions, **kwargs)

    def forward(self, obs, goal, q_target=False, pi_target=False):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        q_net = self.critic_targ if q_target else self.critic
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        return q_net(inputs, pis), pis

    def f(self, obses, actions):
        obses = self._preprocess_obs(obses)
        obses = self._process_obs(obses)
        actions = self.to_tensor(actions)
        return self.critic.f_embed(obses, actions)

    def fsag_f(self, obses, actions, goals):
        obses = self._preprocess_obs(obses)
        obses = self._process_obs(obses)
        goals = self._preprocess_goal(goals)
        goals = self._process_goal(goals)
        actions = self.to_tensor(actions)
        return self.critic.f_embed(obses, actions, goals)

    def phi(self, f_embeds, goals):
        goals = self._preprocess_goal(goals)
        goals = self._process_goal(goals)
        return self.critic.phi_embed(f_embeds, goals)

    def sym_phi(self, goals):
        goals = self._preprocess_goal(goals)
        goals = self._process_goal(goals)
        return self.critic.sym_phi_embed(goals)

    def fsag_phi(self, goals):
        goals = self._preprocess_goal(goals)
        goals = self._process_goal(goals)
        return self.critic.fsag_phi_embed(goals)

    def state_phi(self, obses, goals):
        obses = self._preprocess_obs(obses)
        obses = self._process_obs(obses)
        goals = self._preprocess_goal(goals)
        goals = self._process_goal(goals)
        return self.critic.state_phi_embed(obses, goals)

    def target_update(self):
        net_utils.target_soft_update(source=self.actor, target=self.actor_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic, target=self.critic_targ, polyak=self.args.polyak)

    def normalizer_update(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        self.o_normalizer.update(obs)
        self.g_normalizer.update(goal)
        self.o_normalizer.recompute_stats()
        self.g_normalizer.recompute_stats()

    def state_dict(self):
        return {'actor': self.actor.state_dict(),
                'actor_targ': self.actor_targ.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_targ': self.critic_targ.state_dict(),
                'o_normalizer': self.o_normalizer.state_dict(),
                'g_normalizer': self.g_normalizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_targ.load_state_dict(state_dict['actor_targ'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_targ.load_state_dict(state_dict['critic_targ'])
        self.o_normalizer.load_state_dict(state_dict['o_normalizer'])
        self.g_normalizer.load_state_dict(state_dict['g_normalizer'])
