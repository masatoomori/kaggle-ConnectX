import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
import torch.nn.functional as F


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512) -> None:
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Simple, one convolutional layer with batch normalization. Kernel of
        # size 4 should be good because for detection of 4 marks in row.
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=observation_space.shape[0],
                out_channels=64,
                kernel_size=4,
                stride=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.BatchNorm1d(features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)

        return x
