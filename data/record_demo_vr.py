"""
This file is a demo collection script that you can use on the realworld franka
in order to collect demos for imitation learning.
"""

import gymnasium as gym
from utils.oculus import VRController
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from gymnasium.spaces import flatten

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    SERLObsWrapper
)


def _flatten_obs(obs, env):
    obs = {
        "state": flatten(env.observation_space["state"], obs["state"]),
        **(obs["images"]),
    }
    return obs


if __name__ == "__main__":

    env = gym.make("FrankaPegInsert-Vision-v0")
    env = GripperCloseEnv(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    # env = SERLObsWrapper(env)
    obs, _ = env.reset()


    transitions = []
    success_count = 0
    # Change this to the number of demos you want to record
    success_needed = 0
    pbar = tqdm(total=success_needed)
    controller = VRController()
    current_trajectory = []

    while success_count < success_needed:
        action = controller.forward(obs)[:6]
        next_obs, rew, done, truncated, info = env.step(action=action)
        actions = action
        # print(actions)
        transition = copy.deepcopy(
            dict(
                observations=_flatten_obs(obs, env),
                actions=actions,
                next_observations=_flatten_obs(next_obs, env),
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        current_trajectory.append(transition)

        obs = next_obs

        if done:
            obs, _ = env.reset()
            if rew > 0:
                success_count += 1
                pbar.update(1)
                transitions.extend(current_trajectory)
            current_trajectory = []

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"peg_insert_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")
