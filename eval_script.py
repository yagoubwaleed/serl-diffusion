import collections
from dataclasses import dataclass

import gymnasium as gym
import hydra
import numpy as np
import torch
from tqdm.auto import tqdm

from diffusion_policy.policy import DiffusionPolicy
from diffusion_policy.configs import ExperimentHydraConfig, DiffusionModelRunConfig
from diffusion_policy.make_networks import instantiate_model_artifacts
from diffusion_policy.dataset import normalize_data, unnormalize_data
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    Quat2EulerWrapper,
    SERLObsWrapper
)


@dataclass
class EvalConfig:
    hydra: ExperimentHydraConfig = ExperimentHydraConfig()
    checkpoint_path: str = "${hydra:runtime.cwd}/outputs/checkpoint_w_60_trajectories.pt"
    max_steps: int = 100
    num_eval_episodes: int = 20

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)


def process_obs(obs, nets, device):
    # This function processes the observation such that they can just be fed into the model.
    # It should return a dictionary with the following keys
    # 'embed': The image embeddings
    # 'state': The state of the environment
    # You can change how you get this information depending on the environment.

    state = obs['state']
    with torch.no_grad():
        im1 = torch.tensor(obs['wrist_1'], dtype=torch.float32).permute(2, 0, 1).to(device)
        im2 = torch.tensor(obs['wrist_2'], dtype=torch.float32).permute(2, 0, 1).to(device)
        im_stack = torch.stack([im1, im2], dim=0)
        images = nets['vision_encoder'](im_stack).cpu().flatten().numpy()
    return {
        'embed': images,
        'state': state
    }


def run_one_eval(env: gym.Env, nets: torch.nn.Module, config: DiffusionModelRunConfig, stats, noise_scheduler, device,
                 max_steps: int) -> bool:
    policy = DiffusionPolicy(config, nets, noise_scheduler, stats, device)
    # get first observation
    obs, _ = env.reset()
    policy.add_obs(obs)

    # save visualization and rewards
    rewards = list()
    done = False
    step_idx = 0

    while not done:
        action = policy.get_action()

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _, info = env.step(action[i])
            # save observations
            policy.add_obs(obs)
            # and reward/vis
            rewards.append(reward)

            # update progress bar
            step_idx += 1
            if step_idx > max_steps:
                return False
            if done:
                if reward > 0:
                    return True
                return False


@hydra.main(version_base=None, config_name="eval_config")
def main(cfg: EvalConfig):
    checkpoint = torch.load(cfg.checkpoint_path, map_location='cuda')
    diff_run_config: DiffusionModelRunConfig = checkpoint['config']

    nets, noise_scheduler, device = instantiate_model_artifacts(diff_run_config, model_only=True)
    nets.load_state_dict(checkpoint['state_dict'])
    print('Pretrained weights loaded.')
    stats = checkpoint['stats']

    env = gym.make(
        "FrankaPushing-Vision-v0",
        fake_env=False,
    )
    env = GripperCloseEnv(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    successes = 0
    for _ in tqdm(range(cfg.num_eval_episodes), desc='Evaluating'):
        succeeded = run_one_eval(env=env, nets=nets, config=diff_run_config, stats=stats, noise_scheduler=noise_scheduler,
                     device=device, max_steps=cfg.max_steps)
        if succeeded:
            successes += 1
    # Round to the 3rd decimal place
    success_rate = round(successes / cfg.num_eval_episodes, 3)
    print(f'Success rate: {success_rate}')


if __name__ == "__main__":
    main()
