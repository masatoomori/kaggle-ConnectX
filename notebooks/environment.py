import numpy as np
import gymnasium as gym
from kaggle_environments import make, evaluate

# https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/connectx

REWARD_WIN = 1
REWARD_LOSE = -1
REWARD_INVALID = -10

RANDOM_SEED = 0


class ConnectFourGym(gym.Env):
    def __init__(self, opponent='random', switch_prob=0.5):
        self.env = make("connectx", debug=True)
        self.switch_prob = switch_prob
        self.agents = [None, opponent]
        self.trainer = self.env.train(self.agents)

        self.config = self.env.configuration
        # PyTorch Conv2d expect 4 dimensional data
        # (nSamples x nChannels x Height x Width)
        self.board_template = (1, self.config.rows, self.config.columns)

        # 報酬のレンジを設定
        self.reward_range = (REWARD_INVALID, REWARD_WIN)

        self.action_space = gym.spaces.Discrete(self.config.columns)
        self.observation_space = gym.spaces.Box(
            # lowとhighの間の値をとる
            low=0,
            high=2,
            shape=self.board_template,
            dtype=int
        )

    def switch_starting_positions(self):
        """
        ゲームの開始位置を交換するためのメソッドです。
        このメソッドを呼び出すことで、エージェント（プレイヤー）の開始位置が反転されます。

        具体的には、このメソッドは以下の処理を行います：
        1. agentsリストの要素を反転させることで、self.agents[0]とself.agents[1]の順序を入れ替えます。
           これにより、ゲームの開始時にプレイヤーの役割が交換されます。
        2. self.trainerを新しいagentsリストに基づいて再設定します。
           env.train()メソッドは、指定されたエージェントの組み合わせでトレーニング用の環境を作成します。したがって、self.trainerを再設定することで、新しい開始位置のプレイヤーの組み合わせに基づいてトレーニング環境が再構築されます。

        このメソッドの役割は、ランダムにゲームの開始位置を変更することで、トレーニングデータの多様性を増やすことです。
        ゲームの開始位置が異なると、エージェントが異なる状況に対して学習することになります。
        その結果、より汎化性の高いモデルが得られる可能性があります。
        """
        self.agents = self.agents[::-1]
        self.trainer = self.env.train(self.agents)

    def reset(self, seed=RANDOM_SEED):
        if np.random.random() < self.switch_prob:
            self.switch_starting_positions()

        self.observation = self.trainer.reset()
        self.board = np.array(self.observation['board']).reshape(self.board_template)
        return self.board, {}

    def update_reward(self, prev_reward, is_done):
        if prev_reward == REWARD_WIN:
            return REWARD_WIN
        elif is_done:
            return REWARD_LOSE
        else:
            return 1 / (self.config.rows * self.config.columns)

    def step(self, action):
        is_valid = (self.observation['board'][int(action)] == 0)

        if is_valid:
            self.observation, reward, is_done, info = self.trainer.step(int(action))
            reward = self.update_reward(reward, is_done)
        else:
            reward, is_done, info = REWARD_INVALID, True, {}    # 妥当なアクションでない場合は最大限の罰を与えて終了
        terminated = truncated = is_done
        self.board = np.array(self.observation['board']).reshape(self.board_template)
        return self.board, reward, terminated, truncated, info
