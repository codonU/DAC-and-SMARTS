#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from ..utils import *

try:
    # import roboschool
    pass
except ImportError:
    pass

# SMARTS continuous_space 中的ACTION_SPACE
# ==================================================
# Continous Action Space
# throttle, brake, steering
# ==================================================

ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)

# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(
    env_id, seed, rank, episode_life=True,      # 原来跟在rank后面
    # 现在的设计模式， 在env_id里面以字典包含所有变量
    # SMARTS
    # scenario_path,
    # agent_specs,
    # # set headless to false if u want to use envision
    # headless=False,
    # visdom=False,
    # #   seed=42,
    # AGENT_ID = "AGENT-007",
    ):
    def _thunk():
        
        # SMARTS
        if type(env_id) == dict:
            dic = env_id
            env_name = dic['name']
            env_scenario_path = dic['scenario_path']
            env_agent_spec = dic['agent_spec']
            # env_headless = dic['headless']
            # env_visdom = dic['visdom']
            env_AGENT_ID = dic['AGENT_ID']
            env = gym.make(
                env_name,
                scenarios=env_scenario_path,
                agent_specs={env_AGENT_ID: env_agent_spec},
                # set headless to false if u want to use envision
                # headless=env_headless,
                # visdom=env_visdom,
                seed=42,
            )
            env = SMARTSWrapper(env, env_AGENT_ID)
            return env

        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env
    return _thunk

def make_env_backup(env_id, seed, rank, episode_life=True):
    def _thunk():
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env

    return _thunk


class SMARTSWrapper(gym.Wrapper):
    def __init__(self, env, AGENT_ID):
        gym.Wrapper.__init__(self, env)
        self.agent_id = AGENT_ID
        obs_origin = self.env.reset()

        # self.observation_space
        self.observation_space = self.cal_gym_dic_dim(obs_origin)
        # self.observation_space = np.zeros(self.observation_space)
        # numpy数组可能在后面多环境的包装中不适用
        # 用gym原生Box试一下
        # limit_arr 是按照continuous中observation的具体限制设置的
        limit_arr = np.full((self.observation_space), 1e10)
        limit_arr[1] = 1.0      # heading_errors
        self.observation_space = gym.spaces.Box(
            low=-limit_arr, high=limit_arr, dtype=np.float32
        )
        self.env.observation_space = self.observation_space

        # self.action_space
        self.action_space = ACTION_SPACE
        self.env.action_space = ACTION_SPACE
        # print(self.env.action_space)
    
    def step(self, action):
        origin_obs, reward, done, info = self.env.step({self.agent_id: action})

        # obs = obs[self.agent_id]
        obs = self.concat_obs(origin_obs)

        reward = reward[self.agent_id]
        print("step reward:", reward)
        done = done[self.agent_id]

        info = info[self.agent_id]
        # info在每个AGENT下的内容
        # {'env_obs': 一些全局信息， 'goal_distance': , 'score'}
        if not done:
            info['episodic_return'] = None
        else:
            info['episodic_return'] = info.pop('score')
            # 在这里增加记录平均速度、平均里程等信息到info
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.concat_obs(obs)

    def concat_obs(self, origin_obs):
        """以continues中的agent——interface构建

        OBSERVATION_SPACE = gym.spaces.Dict(
            {
                # To make car follow the waypoints
                # distance from lane center
                "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                # relative heading angle from 10 waypoints in 50 forehead waypoints
                "heading_errors": gym.spaces.Box(low=-1.0, high=1.0, shape=(10,)),
                # Car attributes
                # ego speed
                "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                # ego steering
                "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                # To make car learn to slow down, overtake or dodge
                # distance to the closest car in each lane
                "lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
                # time to collide to the closest car in each lane
                "lane_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
                # ego lane closest social vehicle relative speed
                "closest_lane_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                # distance to the closest car in possible intersection direction
                "intersection_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                # time to collide to the closest car in possible intersection direction
                "intersection_distance": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                # intersection closest social vehicle relative speed
                "closest_its_nv_rel_speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                # intersection closest social vehicle relative position in vehicle heading coordinate
                "closest_its_nv_rel_pos": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),
            }
        )
        """
        obs_dic = origin_obs[self.agent_id]
        obs = np.zeros(self.observation_space.shape[0])
        idx = 0
        for key in obs_dic.keys():
            leng = obs_dic[key].shape[0]
            obs[idx: idx + leng] = obs_dic[key]
            idx += leng
        return obs

    def cal_gym_dic_dim(self, origin_obs):
        """计算gym.dic中各个变量共有多少维
        """
        if type(origin_obs) != dict:
            raise "type error, cal_gym_dic_dim should get dic"
        if len(origin_obs.keys()) == 1:
            origin_obs = origin_obs[self.agent_id]
        dim = 0
        for key in origin_obs.keys():
            # print(origin_obs[key].shape)
            dim += origin_obs[key].shape[0]
        return dim


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class PaddingObsWrapper(gym.Wrapper):
    def __init__(self, env, domain, task):
        gym.Wrapper.__init__(self, env)
        self.domain = domain
        self.task = task
        if self.domain == 'fish' and self.task == 'upright':
            # make upright compatible with swim
            self.observation_space = Box(
                -float('inf'),
                float('inf'),
                (24, ),
                dtype=np.float32,
            )

    def pad_obs(self, obs):
        if self.domain == 'fish' and self.task == 'upright':
            new_obs = np.zeros((24, ))
            new_obs[:8] = obs[:8]
            new_obs[11:] = obs[8:]
            return new_obs
        else:
            return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.pad_obs(obs), reward, done, info

    def reset(self):
        return self.pad_obs(self.env.reset())


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

    def close(self):
        return


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name
        self.observation_space = self.env.observation_space
        # print(self.observation_space)
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
