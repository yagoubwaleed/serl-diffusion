import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class PushingEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": "123622270810",
        "wrist_2": "130322270807",
    }
    TARGET_POSE = np.array([
        0.42665713589860976,
        0.01697357310341893,
        0.030485965848999902,
        3.1130148, 0.0956483, -0.0854985])
    RESET_POSE = TARGET_POSE + np.array([-0.06, 0.0, 0.0, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
    ACTION_SCALE = np.array([0.1, 0.1, 1])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.1
    RANDOM_RZ_RANGE = np.pi / 6
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - 1,
            TARGET_POSE[1] - 1,
            TARGET_POSE[2],
            TARGET_POSE[3] - np.pi / 2,
            TARGET_POSE[4] - np.pi / 2,
            TARGET_POSE[5] - np.pi / 2,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + 1,
            TARGET_POSE[1] + 1,
            TARGET_POSE[2] + 1,
            TARGET_POSE[3] + np.pi / 2,
            TARGET_POSE[4] + np.pi / 2,
            TARGET_POSE[5] + np.pi / 2,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.008,
        "translational_clip_y": 0.008,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.008,
        "translational_clip_neg_y": 0.008,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.07,
        "rotational_clip_y": 0.07,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.07,
        "rotational_clip_neg_y": 0.07,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }
    MAX_EPISODE_STEPS = 100000