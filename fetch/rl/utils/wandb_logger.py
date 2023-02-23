import wandb
from rl import Args

class WandbLogger(object):

  def __init__(self):
    wandb.init(project="bvn_fetch",
      config=vars(Args),
    )

  def start(self):
    pass

  def store(self):
    pass

  def store_key_value(self, k, v):
    pass

  def save_video(self, frames, path):
    pass