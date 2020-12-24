import gym
import numpy as np


env = gym.make("CartPole-v1")
print(env.observation_space)
print(env.observation_space.shape)
state_dim = int(np.prod(env.observation_space.shape))
print(state_dim)