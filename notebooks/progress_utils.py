from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self.pbar = pbar

    def _on_step(self):
        self.pbar.n = self.num_timesteps
        self.pbar.update(0)


class ProgressBarManager(object):
    def __init__(self, total_timesteps):
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):
        self.pbar = tqdm(total=self.total_timesteps)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
