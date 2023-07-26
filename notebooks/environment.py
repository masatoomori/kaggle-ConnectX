import numpy as np
import gym
from kaggle_environments import make, evaluate

# https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/connectx


class ConnectFourGym(gym.Env):
    def __init__(self, agent2='random'):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.row = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns

        self.action_space = gym.spaces.Discrete(self.columns)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, self.row, self.columns), dtype=np.float)

        # 報酬のレンジを設定
        self.reward_range = (-10., 1.)
        # StableBaselinesは下記が設定さていないとエラーになる
        self.space = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        # TODO: 意味を調べる
        return np.array(self.obs['board']).reshape(1, self.row, self.columns) / 2

    def update_reward(self, previous_reward: float, is_done: bool) -> float:
        if previous_reward == 1:    # エージェントがすでに勝っている場合
            return 1.
        elif is_done:   # エージェントが負けた場合
            return -1.
        else:
            return 1 / (self.row * self.columns)    # 少しずつ減らす？
