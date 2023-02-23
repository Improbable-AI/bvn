
from params_proto.neo_proto import PrefixProto

class Args(PrefixProto):
    debug = False
    agent_type = 'ddpg'
    env_name = 'FetchReach-v1'
    n_epochs = 50
    n_cycles = 50
    n_batches = 40
    save_interval = 5
    seed = 123
    n_workers = 1
    replay_strategy = 'future'
    clip_return = 50
    save_dir = 'save_models'
    noise_eps = 0.2
    random_eps = 0.3
    buffer_size = int(1e6)
    replay_k = 4
    clip_obs = 200
    batch_size = 256
    gamma = 0.98
    action_l2 = 1
    lr_actor = 0.001
    lr_critic = 0.001
    polyak = 0.95
    n_test_rollouts = 10
    clip_range = 5
    demo_length = 20
    cuda = False
    num_rollouts_per_mpi = 2

    checkpoint_freq = 10
    critic_type = 'td'

    n_hids = 3
    hid_size = 256

    clip_inputs = True
    normalize_inputs = True

class MetricArgs(PrefixProto):
    metric_embed_dim = 16
    metric_norm_ord = 2