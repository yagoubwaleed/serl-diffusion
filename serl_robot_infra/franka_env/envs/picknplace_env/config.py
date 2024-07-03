import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class PickNPlaceConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""
    SERVER_URL: str = "http://127.0.0.1:5000/"
    MAX_EPISODE_STEPS = 200
    REALSENSE_CAMERAS = {
        "wrist_1": "123622270810",
        "wrist_2": "130322270807",
    }
    TARGET_POSE = None
    RESET_POSE = np.array(
        [
            0.2945907180366626,
            0.434584304953815,
            0.13751238794480508,
            3.157248,
            -0.09709,
            0.1721946
        ]
    )
    REWARD_THRESHOLD = None
    ACTION_SCALE = np.array([.5, 0.5, 1])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = np.pi / 4

    ABS_POSE_LIMIT_LOW = np.array(
        [
            0.2472017308341049,
            0.19075104894248351,
            0.021,
            RESET_POSE[3] - 0.1,
            RESET_POSE[4] - 0.1,
            RESET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )

    ABS_POSE_LIMIT_HIGH = np.array(
        [
            0.5792640702964066,
            0.6341936890734797,
            0.1410515521524619,
            RESET_POSE[3] + 0.1,
            RESET_POSE[4] + 0.1,
            RESET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )

    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.003,
        "translational_clip_y": 0.003,
        "translational_clip_z": 0.003,
        "translational_clip_neg_x": 0.003,
        "translational_clip_neg_y": 0.003,
        "translational_clip_neg_z": 0.003,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
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