from gymnasium.envs.registration import register
import numpy as np

register(
    id="FrankaEnv-Vision-v0",
    entry_point="franka_env.envs:FrankaEnv",
    max_episode_steps=100,
)

register(
    id="FrankaPickNPlace-Vision-v0",
    entry_point="franka_env.envs.picknplace_env:FrankaPickNPlace",
    max_episode_steps=800,
)

register(
    id="FrankaPushing-Vision-v0",
    entry_point="franka_env.envs:pushing_env:FrankaPushing",
    max_episode_steps=100,
)
register(
    id="FrankaPegInsert-Vision-v0",
    entry_point="franka_env.envs.peg_env:FrankaPegInsert",
    max_episode_steps=100,
)

register(
    id="FrankaPCBInsert-Vision-v0",
    entry_point="franka_env.envs.pcb_env:FrankaPCBInsert",
    max_episode_steps=100,
)

register(
    id="FrankaCableRoute-Vision-v0",
    entry_point="franka_env.envs.cable_env:FrankaCableRoute",
    max_episode_steps=100,
)
