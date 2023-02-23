import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

def mlp(sizes):
    layers = []
    n_layer = len(sizes) - 1
    if n_layer <= 0:
        return nn.Identity()
    for j in range(n_layer):
        if j == n_layer - 1:
            activ = nn.Identity
        else:
            activ = nn.ReLU
        layers += [
            nn.Linear(sizes[j], sizes[j + 1]),
            activ()]
    return nn.Sequential(*layers)


# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class critic(nn.Module):
    def __init__(self, env_params, args, *_, **kwargs):
        super(critic, self).__init__()
        self.act_limit = env_params['action_max']

        input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.net = mlp(
            [input_dim] + [args.hid_size] * args.n_hids + [1])

    def forward(self, x, actions):
        q_inputs = torch.cat([x, actions / self.act_limit], dim=-1)
        q_values = self.net(q_inputs)
        
        return q_values

class fullrank_two_stream_critic(nn.Module):
    def __init__(self, env_params, args, metric_args):
        super().__init__()
        self.max_action = env_params['action_max']
        self.env_params = env_params
        self.args = args
        self.metric_args = metric_args

        self.f_fc1 = nn.Linear(env_params['obs'] + env_params['action'], 174)
        self.f_fc2 = nn.Linear(174, 174)
        self.f_fc3 = nn.Linear(174, 174)
        self.f_out = nn.Linear(174, args.metric_embed_dim)

        self.phi_fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 174)
        self.phi_fc2 = nn.Linear(174, 174)
        self.phi_fc3 = nn.Linear(174, 174)
        self.phi_out = nn.Linear(174, args.metric_embed_dim)

    def forward(self, x, actions):
        obs, goal = x[:, :self.env_params['obs']], x[:, self.env_params['obs']:]
        x = torch.cat([obs, actions / self.max_action], dim=1)
        x = F.relu(self.f_fc1(x))
        x = F.relu(self.f_fc2(x))
        x = F.relu(self.f_fc3(x))
        x = self.f_out(x)

        y = torch.cat([obs, goal], dim=1)
        y = F.relu(self.phi_fc1(y))
        y = F.relu(self.phi_fc2(y))
        y = F.relu(self.phi_fc3(y))
        y = self.phi_out(y)

        q_value = self._reduce_embeds(x, y)

        return q_value

    def _reduce_embeds(self, f_embeds, goal_embeds):
        raise NotImplemented

class lowrank_two_stream_critic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.max_action = env_params['action_max']
        self.env_params = env_params
        self.args = args
        self.metric_args = metric_args

        self.f_fc1 = nn.Linear(env_params['obs'] + env_params['action'], 174)
        self.f_fc2 = nn.Linear(174, 174)
        self.f_fc3 = nn.Linear(174, 174)
        self.f_out = nn.Linear(174, args.metric_embed_dim)

        self.phi_fc1 = nn.Linear(env_params['goal'], 174)
        self.phi_fc2 = nn.Linear(174, 174)
        self.phi_fc3 = nn.Linear(174, 174)
        self.phi_out = nn.Linear(174, args.metric_embed_dim)

    def forward(self, x, actions):
        obs, goal = x[:, :self.env_params['obs']], x[:, self.env_params['obs']:]
        x = torch.cat([obs, actions / self.max_action], dim=1)
        x = F.relu(self.f_fc1(x))
        x = F.relu(self.f_fc2(x))
        x = F.relu(self.f_fc3(x))
        x = self.f_out(x)

        y = torch.cat([goal], dim=1)
        y = F.relu(self.phi_fc1(y))
        y = F.relu(self.phi_fc2(y))
        y = F.relu(self.phi_fc3(y))
        y = self.phi_out(y)

        q_value = self._reduce_embeds(x, y)

        return q_value

    def _reduce_embeds(self, f_embeds, goal_embeds):
        raise NotImplemented

class dot_critic(object):

    def _reduce_embeds(self, f_embeds, goal_embeds):
        n_batch, latent_dim = f_embeds.shape
        return torch.bmm(f_embeds.view(n_batch, 1, latent_dim), goal_embeds.view(n_batch, latent_dim, 1)).view(n_batch, 1)

class norm_critic(object):

    def _reduce_embeds(self, f_embeds, goal_embeds):
        n_batch, latent_dim = f_embeds.shape
        return -torch.linalg.norm(f_embeds - goal_embeds, dim=-1, ord=self.metric_args.metric_norm_ord).view(n_batch, 1)

class fullrank_dot_critic(dot_critic, fullrank_two_stream_critic):
    pass

class fullrank_norm_critic(norm_critic, fullrank_two_stream_critic):
    pass

class lowrank_dot_critic(dot_critic, lowrank_two_stream_critic):
    pass

class lowrank_norm_critic(norm_critic, lowrank_two_stream_critic):
    pass

