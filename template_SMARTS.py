from deep_rl import *
import subprocess
import argparse
from pathlib import Path

import gym

from smarts.core.agent import AgentPolicy
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType, DiscreteAction

import warnings
warnings.filterwarnings("ignore")


def batch_mujoco():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        'dm-acrobot-swingup',
        'dm-acrobot-swingup_sparse',
        'dm-ball_in_cup-catch',
        'dm-cartpole-swingup',
        'dm-cartpole-swingup_sparse',
        'dm-cartpole-balance',
        'dm-cartpole-balance_sparse',
        'dm-cheetah-run',
        'dm-finger-turn_hard',
        'dm-finger-spin',
        'dm-finger-turn_easy',
        'dm-fish-upright',
        'dm-fish-swim',
        'dm-hopper-stand',
        'dm-hopper-hop',
        'dm-humanoid-stand',
        'dm-humanoid-walk',
        'dm-humanoid-run',
        'dm-manipulator-bring_ball',
        'dm-pendulum-swingup',
        'dm-point_mass-easy',
        'dm-reacher-easy',
        'dm-reacher-hard',
        'dm-swimmer-swimmer15',
        'dm-swimmer-swimmer6',
        'dm-walker-stand',
        'dm-walker-walk',
        'dm-walker-run',
    ]

    games = ['HalfCheetah-v2', 'Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Reacher-v2']

    params = []

    # for game in games:
    #     for r in range(2):
    #         for num_o in [4]:
    #             for learning in ['all']:
    #                 for opt_ep in [5, 10]:
    #                     for entropy_weight in [0, 0.01]:
    #                         params.append([a_squared_c_ppo_continuous,
    #                                        dict(game=game, run=r, learning=learning, num_o=num_o, opt_ep=opt_ep,
    #                                             entropy_weight=entropy_weight, tasks=False)])
    #         params.append([ppo_continuous, dict(game=game, run=r, tasks=False)])

    for game in games:
        for r in range(1):
            # params.append([a_squared_c_ppo_continuous, dict(game=game, run=r, tasks=False, remark='ASC-PPO', gate=nn.Tanh())])
            # params.append([a_squared_c_a2c_continuous, dict(game=game, run=r, tasks=False, remark='ASC-A2C', gate=nn.Tanh())])
            # params.append([ppo_continuous, dict(game=game, run=r, tasks=False, remark='PPO', gate=nn.Tanh())])
            # params.append([a2c_continuous, dict(game=game, run=r, tasks=False, remark='A2C',gate=nn.Tanh())])
            # params.append([ahp_ppo_continuous, dict(game=game, run=r, tasks=False, remark='AHP', gate=nn.Tanh())])

            params.append([oc_continuous, dict(game=game, run=r, tasks=False, remark='OC', gate=nn.Tanh())])
            params.append([ppoc_continuous, dict(game=game, run=r, tasks=False, remark='PPOC', gate=nn.Tanh())])
            params.append([fixed_std_oc, dict(game=game, run=r, tasks=False, remark='Fixed_OC', gate=nn.Tanh())])
            # params.append([a_squared_c_a2c_continuous,
            #                dict(game=game, run=r, tasks=False, remark='ASC-A2C', gate=nn.Tanh(), num_workers=4)])
            # params.append(
            #     [oc_continuous, dict(game=game, run=r, tasks=False, remark='OC', gate=nn.Tanh(), num_workers=4)])
            # params.append(
            #     [iopg_continuous, dict(game=game, run=r, tasks=False, remark='IOPG', gate=nn.Tanh(), num_workers=4)])
            # params.append([iopg_continuous, dict(game=game, run=r, tasks=False, remark='IOPG', gate=nn.Tanh())])

    # params = []
    # for r in range(2):
    #     for num_o in [2, 4, 8]:
    #         for beta_w in [0, 0.01, 0.1]:
    #             params.append([a_squared_c_ppo_continuous, dict(game='dm-cheetah', run=r, tasks=True, remark='vis',
    #                                                             num_o=num_o, beta_weight=beta_w)])

    for i in range(len(params)):
        algo, param = params[i]
        algo(**param)
    # algo, param = params[cf.i]
    # algo(**param)
    # a_squared_c_ppo_continuous(**params[cf.i])
    # exit()


def batch_dm():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    games = [
        'dm-acrobot-swingup',
        'dm-acrobot-swingup_sparse',
        'dm-ball_in_cup-catch',
        'dm-cartpole-swingup',
        'dm-cartpole-swingup_sparse',
        'dm-cartpole-balance',
        'dm-cartpole-balance_sparse',
        'dm-cheetah-run',
        'dm-finger-turn_hard',
        'dm-finger-spin',
        'dm-finger-turn_easy',
        'dm-fish-upright',
        'dm-fish-swim',
        'dm-hopper-stand',
        'dm-hopper-hop',
        'dm-humanoid-stand',
        'dm-humanoid-walk',
        'dm-humanoid-run',
        'dm-manipulator-bring_ball',
        'dm-pendulum-swingup',
        'dm-point_mass-easy',
        'dm-reacher-easy',
        'dm-reacher-hard',
        'dm-swimmer-swimmer15',
        'dm-swimmer-swimmer6',
        'dm-walker-stand',
        'dm-walker-walk',
        'dm-walker-run',
    ]

    games = ['dm-cartpole-b', 'dm-reacher', 'dm-fish', 'dm-cheetah', 'dm-walker-1', 'dm-walker-2']
    # games = []
    # games = ['dm-cheetah', 'dm-walker-2']
    # games = ['dm-walker-1']

    params = []
    # for game in games:
    #     for r in range(2):
    #         for num_o in [4]:
    #             for learning in ['all']:
    #                 for opt_ep in [5, 10]:
    #                     for entropy_weight in [0, 0.01]:
    #                         params.append([a_squared_c_ppo_continuous,
    #                                        dict(game=game, run=r, learning=learning, num_o=num_o, opt_ep=opt_ep,
    #                                             entropy_weight=entropy_weight, tasks=False)])
    #         params.append([ppo_continuous, dict(game=game, run=r, tasks=False)])

    for game in games:
        for r in range(1):
            # params.append([a_squared_c_ppo_continuous, dict(game=game, run=r, tasks=True, remark='ASC-PPO')])
            # params.append([ppo_continuous, dict(game=game, run=r, tasks=True, remark='PPO')])
            # params.append([ahp_ppo_continuous, dict(game=game, run=r, tasks=True, remark='AHP')])

            params.append([ppoc_continuous, dict(game=game, run=r, tasks=True, remark='PPOC')])
            params.append([oc_continuous, dict(game=game, run=r, tasks=True, remark='OC')])
            params.append([fixed_std_oc, dict(game=game, run=r, tasks=True, remark='OC')])
            # params.append([ahp_ppo_continuous, dict(game=game, run=r, tasks=True, remark='AHP')])
            # params.append([iopg_continuous, dict(game=game, run=r, tasks=True, remark='IOPG')])
            # params.append([oc_continuous, dict(game=game, run=r, tasks=True, remark='OC', num_workers=4)])
            # params.append([iopg_continuous, dict(game=game, run=r, tasks=True, remark='IOPG', num_workers=4)])

    # params = []
    # for r in range(3):
    #     for game in games:
    #         # for num_o in [2, 4, 8]:
    #         # for algo in [a_squared_c_ppo_continuous, ahp_ppo_continuous]:
    #         params.append([a_squared_c_ppo_continuous, dict(game=game, run=r, tasks=True, remark='ASC', log_level=1,
    #                                                         save_interval=int(1e6 / 2048) * 2048)])
    #         params.append([ahp_ppo_continuous, dict(game=game, run=r, tasks=True, remark='AHP', log_level=1,
    #                                                 save_interval=int(1e6 / 2048) * 2048)])

    for i in range(len(params)):
        algo, param = params[i]
        algo(**param)
    # algo, param = params[cf.i]
    # algo(**param)
    exit()


def set_tasks(config):
    if config.game == 'dm-walker':
        tasks = ['walk', 'run']
    elif config.game == 'dm-walker-1':
        tasks = ['squat', 'stand']
        config.game = 'dm-walker'
    elif config.game == 'dm-walker-2':
        tasks = ['walk', 'backward']
        config.game = 'dm-walker'
    elif config.game == 'dm-finger':
        tasks = ['turn_easy', 'turn_hard']
    elif config.game == 'dm-reacher':
        tasks = ['easy', 'hard']
    elif config.game == 'dm-cartpole-b':
        tasks = ['balance', 'balance_sparse']
        config.game = 'dm-cartpole'
    elif config.game == 'dm-cartpole-s':
        tasks = ['swingup', 'swingup_sparse']
        config.game = 'dm-cartpole'
    elif config.game == 'dm-fish':
        tasks = ['upright', 'downleft']
    elif config.game == 'dm-hopper':
        tasks = ['stand', 'hop']
    elif config.game == 'dm-acrobot':
        tasks = ['swingup', 'swingup_sparse']
    elif config.game == 'dm-manipulator':
        tasks = ['bring_ball', 'bring_peg']
    elif config.game == 'dm-cheetah':
        tasks = ['run', 'backward']
    else:
        raise NotImplementedError

    games = ['%s-%s' % (config.game, t) for t in tasks]
    config.tasks = [Task(g, num_envs=config.num_workers) for g in games]
    config.game = games[0]


# DAC+PPO
def a_squared_c_ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('freeze_v', False)
    kwargs.setdefault('opt_ep', 5)
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    kwargs.setdefault('beta_weight', 0)
    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    # task_fn 先创建了创建任务的函数
    # 然后要先运行一下才能得到state_dim等必须的参数用于创建网络
    # 目前的解决方法是在算法实体里不再用task_fn创建新的任务instance
    # 而是直接在算法实体里 task = config.eval_env

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = config.opt_ep
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    run_steps(ASquaredCPPOAgent(config))


# DAC+A2C
def a_squared_c_a2c_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('freeze_v', False)
    kwargs.setdefault('opt_ep', 5)
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    kwargs.setdefault('num_workers', 16)
    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 5
    config.state_normalizer = MeanStdNormalizer()
    run_steps(ASquaredCA2CAgent(config))


# example.py
# A2C
def a2c_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)

    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    kwargs.setdefault('num_workers', 16)

    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim), critic_body=FCBody(config.state_dim))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    run_steps(A2CAgent(config))


# PPO
def ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOAgent(config))


# OC
def oc_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    # kwargs.setdefault('num_workers', 16)
    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.random_option_prob = LinearSchedule(0.1)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.gradient_clip = 0.5
    config.rollout_length = 5
    config.beta_reg = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.target_network_update_freq = int(1e3)
    run_steps(OCAgent(config))


# determinstic option critic
def fixed_std_oc(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    kwargs.setdefault('num_workers', 16)
    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)

    config.network_fn = lambda: FixedSTDOptionCritic(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.random_option_prob = LinearSchedule(0.1)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.gradient_clip = 0.5
    config.rollout_length = 5
    config.beta_reg = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.target_network_update_freq = int(1e3)
    run_steps(OCAgent(config))


# PPOC
def ppoc_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = Task(config.game)

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.beta_reg = 0.01
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOCAgent(config))


# AHP+PPO
def ahp_ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('opt_ep', 10)
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = config.opt_ep
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    run_steps(AHPPPOAgent(config))


# IOPG
def iopg_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    kwargs.setdefault('num_workers', 16)
    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.state_normalizer = MeanStdNormalizer()
    run_steps(IOPGAgent(config))


def visualize_a_squared_c(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', nn.ReLU())
    kwargs.setdefault('freeze_v', False)
    kwargs.setdefault('opt_ep', 5)
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 2e6)
    kwargs.setdefault('beta_weight', 0)
    config = Config()
    config.merge(kwargs)

    if config.tasks:
        set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = config.opt_ep
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    agent = ASquaredCPPOAgent(config)

    folder = 'data/ASquaredC'
    steps = [999424, 1998848]
    game = kwargs['game']
    for s, t in zip(steps, config.tasks):
        agent.all_options = []
        saved = '%s/ASquaredCPPOAgent-%s-log_level_1-remark_ASC-save_interval_999424-tasks_True-run-0-%d' % (
        folder, game, s)
        agent.load(saved)
        sub_folder = '%s/%s_episode_%d' % (folder, game, s)
        agent.record_episode(sub_folder, t)
        subprocess.run(['ffmpeg', '-i', '%s/%%04d.png' % (sub_folder), '%s.gif' % (sub_folder)])
        with open('%s_options.bin' % (sub_folder), 'wb') as f:
            pickle.dump(agent.all_options, f)


def batch_SMARTS():
    cf = Config()

    # 直接引用默认的agent_spec
    from starter_kit.train_example.utils.continuous_space import agent_spec

    scenario_path = [
        # './dataset_public/all_loop/all_loop_a'
        # './dataset_public/intersection_loop/its_a',
        # './dataset_public/merge_loop/merge_a',s
        # './dataset_public/mixed_loop/its_merge_a',
        # './dataset_public/mixed_loop/roundabout_its_a',
        # './dataset_public/mixed_loop/roundabout_merge_a',
        # './dataset_public/roundabout_loop/roundabout_a',
        # './dataset_public/sharp_loop/sharploop_a',
        './dataset_public/simple_loop/simpleloop_a',
    ]
    scenario_path = [Path(path).absolute() for path in scenario_path]

    games = [
        {
            'name': "smarts.env:hiway-v0",
            'scenario_path': scenario_path,
            'agent_spec': agent_spec,
            # 'headless': False,
            # 'visdom': False,
            'AGENT_ID': "AGENT-007",
        }
    ]

    params = []

    for game in games:
        for r in range(1):
            # params.append([a_squared_c_ppo_continuous, dict(game=game, run=r, tasks=False, remark='ASC-PPO', gate=nn.Tanh())])
            # params.append([a_squared_c_a2c_continuous, dict(game=game, run=r, tasks=False, remark='ASC-A2C', gate=nn.Tanh())])
            # params.append([ppo_continuous, dict(game=game, run=r, tasks=False, remark='PPO', gate=nn.Tanh(), smarts=True)])
            # params.append([a2c_continuous, dict(game=game, run=r, tasks=False, remark='A2C',gate=nn.Tanh(), smarts=True)])
            # params.append([ahp_ppo_continuous, dict(game=game, run=r, tasks=False, remark='AHP', gate=nn.Tanh(), smarts=True)])
            params.append([oc_continuous, dict(game=game, run=r, tasks=False, remark='OC', gate=nn.Tanh(), smarts=True)])
            params.append([ppoc_continuous, dict(game=game, run=r, tasks=False, remark='PPOC', gate=nn.Tanh(), smarts=True)])

            # params.append([a_squared_c_a2c_continuous,
            #                dict(game=game, run=r, tasks=False, remark='ASC-A2C', gate=nn.Tanh(), num_workers=4)])
            # params.append(
            #     [oc_continuous, dict(game=game, run=r, tasks=False, remark='OC', gate=nn.Tanh(), num_workers=4)])
            # params.append(
            #     [iopg_continuous, dict(game=game, run=r, tasks=False, remark='IOPG', gate=nn.Tanh(), num_workers=4)])
            # params.append([iopg_continuous, dict(game=game, run=r, tasks=False, remark='IOPG', gate=nn.Tanh())])

    for i in range(len(params)):
        algo, param = params[i]
        algo(**param)


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()
    set_one_thread()
    select_device(0)

    batch_SMARTS()

    exit()
