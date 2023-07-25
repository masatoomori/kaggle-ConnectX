import gym
from kaggle_environments import make, evaluate

# https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/connectx


class ConnectFourGym(gym.Env):
    def __init__(self, agent2='random'):
        ks_env = make("connectx", debug=True)
