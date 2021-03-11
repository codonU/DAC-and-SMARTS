import argparse
from pathlib import Path

import gym
import numpy as np
from utils.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from utils.saved_model import RLlibTFCheckpointPolicy


def parse_args():
    parser = argparse.ArgumentParser("run simple keep lane agent")
    # env setting
    parser.add_argument("--scenario", "-s", type=str, help="Path to scenario")
    parser.add_argument("--load_path", "-p", type=str, help="path to stored model")
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )

    return parser.parse_args()


def main(args):

    """
    scenario_path = [
        # './dataset_public/all_loop/all_loop_a'
        # './dataset_public/intersection_loop/its_a',
        # './dataset_public/merge_loop/merge_a',
        # './dataset_public/mixed_loop/its_merge_a',
        # './dataset_public/mixed_loop/roundabout_its_a',
        # './dataset_public/mixed_loop/roundabout_merge_a',
        # './dataset_public/roundabout_loop/roundabout_a',
        # './dataset_public/sharp_loop/sharploop_a',
        # './dataset_public/simple_loop/simpleloop_a',
    ]
    """

    args.scenario = './dataset_public/simple_loop/simpleloop_a'
    args.load_path = './result/PPO_RLlibHiWayEnv_0_Continuous/checkpoint_18/checkpoint-18'
    print(Path(args.load_path).absolute())
    print("args.scenario", args.scenario)
    print("args.load_path", args.load_path)
    print(ACTION_SPACE)

    AGENT_ID = "AGENT-007"

    agent_spec.policy_builder = lambda: RLlibTFCheckpointPolicy(
        Path(args.load_path).absolute(),
        "PPO",
        "default_policy",
        OBSERVATION_SPACE,
        ACTION_SPACE,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[Path(args.scenario).absolute()],
        agent_specs={AGENT_ID: agent_spec},
        # set headless to false if u want to use envision
        headless=False,
        visdom=False,
        seed=42,
    )

    agent = agent_spec.build_agent()

    data = np.zeros((30, 3))
    epi = 0
    while True:
        step = 0
        observations = env.reset()
        total_reward = 0.0
        dones = {"__all__": False}

        speed = []
        
        while not dones["__all__"]:
            step += 1
            print(step)
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, info = env.step({AGENT_ID: agent_action})
            total_reward += rewards[AGENT_ID]
            info = info[AGENT_ID]
            speed.append(info['env_obs'].ego_vehicle_state.speed)
        print("Accumulated reward:", total_reward)
        data[epi, 0] = np.average(speed)
        data[epi, 1] = info['env_obs'].distance_travelled
        if info['env_obs'].events.off_road or len(info['env_obs'].events.collisions):
            data[epi, 2] = 1
        epi += 1
        if epi == 30:
            np.save('PPO_smarts.npy', data)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
