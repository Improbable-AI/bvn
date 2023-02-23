import threading

import numpy as np


def sample_her_transitions(buffer, reward_func, batch_size, future_p=1.0):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]

    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

    future_offset = (np.random.uniform(size=batch_size) * (horizon - t_samples)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None)
    batch['r_ag_ag2'] = reward_func(batch['ag2'], batch['ag'], None)
    batch['r_future_ag'] = reward_func(batch['ag2'], batch['future_ag'], None)

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_state_asym_metric_transitions(buffer, reward_func, batch_size, future_p=1.0):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :]
    buffer['ag2'] = buffer['ag'][:, 1:, :]

    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon - 1, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    batch['a2'] = buffer['a'][ep_idxes, t_samples + 1].copy()
    batch['ag3'] = buffer['ag'][ep_idxes, t_samples + 2].copy()

    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

    future_offset = (np.random.uniform(size=batch_size) * (horizon - 1 - t_samples)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['future_ag2'] = buffer['ag2'][ep_idxes, future_t].copy()
    batch['future_ob'] = buffer['ob'][ep_idxes, future_t].copy()
    batch['future_a'] = buffer['a'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None)
    batch['r_ag_ag2'] = reward_func(batch['ag2'], batch['ag'], None)
    batch['r_future_ag'] = reward_func(batch['ag2'], batch['future_ag'], None)
    batch['r_ag3_ag2'] = reward_func(batch['ag2'], batch['ag3'], None)

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_fsag_metric_transitions(buffer, reward_func, batch_size, future_p=1.0):
    # Need s_t, a_t, g_{t+1}, s_{t+1}, a_{t+1}, g_{t+2}, g_{t+k}
    # g_t = ag
    # s_t = o
    # a_t = a
    # g_{t+1} = ag2
    # s_{t+1} = o2
    # g_{t+k} = future_ag
    # R(g_t, g_{t+1}) = r_ag_ag2
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    buffer['o2'] = buffer['ob'][:, 1:, :] # s_{t+1}
    buffer['ag2'] = buffer['ag'][:, 1:, :] # g_{t+1}

    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size) # horizon - 1 cauz we need t+2
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

    future_offset = (np.random.uniform(size=batch_size) * (horizon - t_samples)).astype(int)
    future_t = (t_samples + 1 + future_offset)

    batch['bg'][her_indexes] = buffer['ag'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None)
    batch['r_ag_ag2'] = reward_func(batch['ag2'], batch['ag'], None)
    batch['r_future_ag'] = reward_func(batch['ag2'], batch['future_ag'], None)

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in ['ob', 'ag', 'bg', 'a', 'o2', 'ag2', 'r', 'future_ag', 'offset'])
    return batch

def sample_asym_metric_transitions(buffer, reward_func, batch_size, future_p=1.0):
    assert all(k in buffer for k in ['ob', 'ag', 'bg', 'a'])
    n_trajs = buffer['a'].shape[0]
    horizon = buffer['a'].shape[1]
    
    buffer['ag2'] = buffer['ag'][:, 1:, :]

    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(1, horizon - 1, size=batch_size) # t in [1, H - 1]
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
            
    past_offset = (np.random.uniform(size=batch_size) * (t_samples)).astype(int)
    past_t = (t_samples - 1 - past_offset)
    future_offset = (np.random.uniform(size=batch_size) * (horizon  - 1 - t_samples)).astype(int)
    future_t = (t_samples + 1 + future_offset)    
    batch['future_ag'] = buffer['ag'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()        
    
    # s_{t-k}, a_{t-k}
    batch['ob_past'] = buffer['ob'][ep_idxes, past_t]
    batch['a_past'] = buffer['a'][ep_idxes, past_t]
    
    # s_{t+1}, a_{t+1}
    batch['ob2'] = buffer['ob'][ep_idxes, t_samples + 1]
    batch['a2'] = buffer['a'][ep_idxes, t_samples + 1]
    
    # s_{t+k}, a_{t+k}
    batch['ob_future'] = buffer['ob'][ep_idxes, future_t]
    batch['a_future'] = buffer['a'][ep_idxes, future_t]
    
    batch['r'] = reward_func(batch['ag2'], batch['bg'], None)

    return batch

class Replay:
    def __init__(self, env_params, args, reward_func, name='replay'):
        self.env_params = env_params
        self.args = args
        self.reward_func = reward_func

        self.horizon = env_params['max_timesteps']
        self.size = args.buffer_size // self.horizon

        self.current_size = 0
        self.n_transitions_stored = 0

        self.buffers = dict(ob=np.zeros((self.size, self.horizon + 1, self.env_params['obs'])),
                            ag=np.zeros((self.size, self.horizon + 1, self.env_params['goal'])),
                            bg=np.zeros((self.size, self.horizon, self.env_params['goal'])),
                            a=np.zeros((self.size, self.horizon, self.env_params['action'])))

        self.lock = threading.Lock()
        # self._save_file = str(name) + '_' + str(mpi_utils.get_rank()) + '.pt'

    def store(self, episodes):
        ob_list, ag_list, bg_list, a_list = episodes['ob'], episodes['ag'], episodes['bg'], episodes['a']
        batch_size = ob_list.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers['ob'][idxs] = ob_list.copy()
            self.buffers['ag'][idxs] = ag_list.copy()
            self.buffers['bg'][idxs] = bg_list.copy()
            self.buffers['a'][idxs] = a_list.copy()
            self.n_transitions_stored += self.horizon * batch_size

    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        if self.args.critic_loss_type == 'asym_metric':
            transitions = sample_asym_metric_transitions(temp_buffers, self.reward_func, batch_size, future_p=self.args.future_p)
        elif self.args.critic_loss_type == 'fsag_metric':
            transitions = sample_fsag_metric_transitions(temp_buffers, self.reward_func, batch_size, future_p=self.args.future_p)
        elif self.args.critic_loss_type == 'state_asym_metric':
            transitions = sample_state_asym_metric_transitions(temp_buffers, self.reward_func, batch_size, future_p=self.args.future_p)
        else:
            transitions = sample_her_transitions(temp_buffers, self.reward_func, batch_size, future_p=self.args.future_p)
        return transitions

    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx

    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )

    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']

    def save(self, path="replay/snapshot.pkl"):
        from ml_logger import logger
        logger.save_module(self, path)

    def load(self, path="replay/snapshot.pkl"):
        from ml_logger import logger
        logger.load_module(self, path)
