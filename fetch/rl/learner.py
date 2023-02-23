import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

import os.path as osp
from rl.utils import mpi_utils


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def dict_to_numpy(tensor_dict):
    return {
        k: to_numpy(v) for k, v in tensor_dict.items()
    }


class Learner:
    def __init__(
            self,
            agent,
            args,
            name='learner',
    ):
        self.agent = agent
        from ml_logger import logger
        self.logger = logger
        self.args = args

        self.optim_q = Adam(agent.critic.parameters(), lr=args.lr_critic)
        self.optim_pi = Adam(agent.actor.parameters(), lr=args.lr_actor)

        self._save_file = str(name) + '.pt'

        self.loss_fn = {'l1': nn.SmoothL1Loss(),
                        'l2': nn.MSELoss()}[self.args.critic_loss_fn]

    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        r_ag_ag2 = self.agent.to_tensor(batch['r_ag_ag2'].flatten())
        r_future_ag = self.agent.to_tensor(batch['r_future_ag'].flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())

        with torch.no_grad():
            q_next, _ = self.agent.forward(o2, bg, q_target=True, pi_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)

        q_bg = self.agent.get_qs(o, bg, a)
        loss_q = (q_bg - q_targ).pow(2).mean()

        q_ag2 = self.agent.get_qs(o, ag2, a)
        loss_ag2 = q_ag2.pow(2).mean()

        q_future = self.agent.get_qs(o, future_ag, a)

        loss_critic = loss_q

        return loss_critic

    def actor_loss(self, batch):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']

        a = self.agent.to_tensor(a)

        q_pi, pi = self.agent.forward(o, bg)
        action_l2 = (pi / self.agent.actor.act_limit).pow(2).mean()
        loss_actor = (- q_pi).mean() + self.args.action_l2 * action_l2

        with torch.no_grad():
            self.logger.store(loss_actor=loss_actor.item(),
                              action_l2=action_l2.item(),
                              q_pi=q_pi.mean().numpy(), )

        return loss_actor

    def update(self, batch):
        loss_critic = self.critic_loss(batch)
        self.optim_q.zero_grad()
        loss_critic.backward()
        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_grads(self.agent.critic, scale_grad_by_procs=True)
        self.optim_q.step()

        for i in range(self.args.n_actor_optim_steps):
            loss_actor = self.actor_loss(batch)
            self.optim_pi.zero_grad()
            loss_actor.backward()
            # if mpi_utils.use_mpi():
            #     mpi_utils.sync_grads(self.agent.actor, scale_grad_by_procs=True)
            self.optim_pi.step()

    def target_update(self):
        self.agent.target_update()

    def state_dict(self):
        return dict(
            q_optim=self.optim_q.state_dict(),
            pi_optim=self.optim_pi.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.optim_q.load_state_dict(state_dict['q_optim'])
        self.optim_pi.load_state_dict(state_dict['pi_optim'])

    def save(self, path):
        if mpi_utils.is_root():
            state_dict = self.state_dict()
            save_path = osp.join(path, self._save_file)
            torch.save(state_dict, save_path)

    def load(self, path):
        load_path = osp.join(path, self._save_file)
        state_dict = torch.load(load_path)
        self.load_state_dict(state_dict)
