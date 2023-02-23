import os
from params_proto.neo_hyper import Sweep

if __name__ == '__main__':
    from rl import Args, main

    with Sweep(Args) as sweep:
        Args.gamma = 0.99
        Args.clip_inputs = True
        Args.normalize_inputs = True
        Args.critic_type = 'state_asym_metric'
        Args.critic_loss_type = 'td'

        with sweep.product:
            with sweep.zip:
                Args.env_name = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1']
                Args.n_workers = [2, 8, 16, 20]
                Args.n_epochs = [50, 150, 200, 500]

            Args.seed = [100, 200, 300, 400, 500]
            Args.metric_embed_dim = [16,]

    for i, deps in sweep.items():
      os.environ["ML_LOGGER_ROOT"] = f"{os.getcwd()}/results/bvn/{deps['Args.env_name']}/{deps['Args.seed']}"
      main(deps)
