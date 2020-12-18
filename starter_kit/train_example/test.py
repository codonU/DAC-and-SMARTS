import numpy as np
import gym
from pathlib import Path

from smarts.core.agent import AgentSpec, AgentPolicy
from smarts.core.agent_interface import AgentInterface, OGM, NeighborhoodVehicles
from smarts.core.controllers import ActionSpaceType, DiscreteAction
from smarts.core.utils.episodes import episodes

from utils.discrete_space import observation_adapter, reward_adapter, action_adapter



class NativeDQN(AgentPolicy):
    def __init__(self, action_n):
        self.action_n = action_n

    def act(self, obs):
        return 2
        # return np.random.choice([i for i in range(self.action_n)])


def custom_observation_adapter(env_obs):
    raw_obs = observation_adapter(env_obs)
    obs = np.array([])
    for n, v in raw_obs.items():
        obs = np.concatenate([obs, v])
    return obs, env_obs

def custom_reward_adapter(env_obs, env_reward):
    rew = reward_adapter(env_obs, env_reward)
    return rew

agent_interface = AgentInterface(
    max_episode_steps=None,
    waypoints=True,
    # neighborhood < 60m
    neighborhood_vehicles=NeighborhoodVehicles(radius=60),
    # OGM within 64 * 0.25 = 16
    ogm=OGM(64, 64, 0.25),
    action=ActionSpaceType.Lane,
)

agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=custom_observation_adapter,
    reward_adapter=custom_reward_adapter,
    action_adapter=action_adapter,
    # info_adapter=info_adapter,
    policy_params={"action_n": 4},
    policy_builder=NativeDQN
)

AGENT_ID = "Agent-007"

scenario_path = Path("../dataset_public/all_loop/").absolute()
print(scenario_path)

env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=[str(scenario_path)],
    headless=True,
    agent_specs={AGENT_ID: agent_spec},
)

for episode in episodes(n=100):
    agent = agent_spec.build_agent()
    observations = env.reset()
    episode.record_scenario(env.scenario_log)

    dones = {"__all__": False}
    total_rewards = 0
    # f = open('env_obs', 'w')
    while not dones["__all__"]:
        agent_obs, env_obs = observations[AGENT_ID]
        # print(agent_obs.shape)
        # f.write(str(env_obs))
        # print(env_obs)
        action = agent.act(agent_obs)
        observations, rewards, dones, infos = env.step({AGENT_ID: action})
        print('reward:{}'.format(rewards))
        total_rewards += rewards[AGENT_ID]
        episode.record_step(observations, rewards, dones, infos)
        break
    # print('episode rew:{}'.format(total_rewards))
    

env.close()




