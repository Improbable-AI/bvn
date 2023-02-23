import sys
from params_proto.neo_proto import ParamsProto, Flag, Proto, PrefixProto


class Args(PrefixProto):
    debug = True if "pydevd" in sys.modules else False
    record_video = True
    train_type = 'online'

    # Critic type
    critic_type = 'td'
    critic_loss_type = 'td'
    critic_loss_fn = 'l1'
    metric_loss_ablation = None
    test_td_compatiability = False
    lamb = 0.01
    metric_embed_dim = 16
    metric_critic_spring_loss_weight = 0.2

    # Actor
    n_actor_optim_steps = 1

    # experimental features
    object_relabel = False

    env_name = "FetchReach-v1"
    test_env_name = None

    seed = 123
    save_dir = "experiments/"
    ckpt_name = ""
    resume_ckpt = ""

    n_workers = 2 if debug else 12
    cuda = Flag("cuda tend to be slower.")
    num_rollouts_per_mpi = 1

    n_epochs = 200
    n_cycles = 10
    optimize_every = 2
    n_batches = 1

    hid_size = 256
    n_hids = 3
    activ = "relu"
    noise_eps = 0.1
    random_eps = 0.2

    buffer_size = 2500000
    future_p = 0.8
    batch_size = 1024

    clip_inputs = Flag("to turn on input clipping")
    clip_obs = Proto(200, dtype=float)

    normalize_inputs = Flag("to normalize the inputs")
    clip_range = Proto(5, dtype=float)

    gamma = 0.98
    clip_return = Proto(50, dtype=float)

    action_l2 = 0.01
    lr_actor = 0.001
    lr_critic = 0.001

    polyak = 0.995
    target_update_freq = 10
    checkpoint_freq = 10

    n_initial_rollouts = 1 if debug else 100
    n_test_rollouts = 15
    demo_length = 20

    logdir = None


def main(deps=None):
    from rl.launcher import launch

    algo = launch(deps)
    algo.run()
