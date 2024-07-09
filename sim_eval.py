import collections
import json
import os
from typing import Callable, Union
import robosuite as suite
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm
from diffusion_policy.policy import DiffusionPolicy
from diffusion_policy.configs import DiffusionModelRunConfig, DatasetConfig
from diffusion_policy.dataset import normalize_data, unnormalize_data
from diffusion_policy.make_networks import instantiate_model_artifacts
from utils.video_recorder import VideoRecorder


@dataclass
class EvalConfig:
    render: bool = False
    video_save_path: Union[str, None] = "outputs/16_epoch_videos"
    max_steps: int = 500
    model_checkpoint: str = "16_epoch_peg.pt"
    sim_json_path: str = "data/square_peg.json"
    num_eval_episodes: int = 20



def main(cfg: EvalConfig):
    metadata = json.load(open(cfg.sim_json_path, "r"))
    kwargs = metadata["env_kwargs"]
    if cfg.render:
        kwargs["has_renderer"] = True
    if cfg.video_save_path is not None:
        os.makedirs(cfg.video_save_path, exist_ok=True)

    env = suite.make(
        env_name=metadata["env_name"],
        **kwargs
    )


    checkpoint = torch.load(cfg.model_checkpoint, map_location='cuda')
    diff_run_config: DiffusionModelRunConfig = checkpoint['config']

    nets, noise_scheduler, device = instantiate_model_artifacts(diff_run_config, model_only=True)
    nets.load_state_dict(checkpoint['state_dict'])
    print('Pretrained weights loaded.')
    stats = checkpoint['stats']
    successes = 0
    for i in tqdm(range(cfg.num_eval_episodes), desc='Evaluating'):
        succeeded = run_one_eval(env=env, nets=nets, config=diff_run_config, stats=stats,
                                 noise_scheduler=noise_scheduler,
                                 device=device, max_steps=cfg.max_steps, render=cfg.render,
                                 save_path=cfg.video_save_path + f"/episode_{i}")
        if succeeded:
            successes += 1
    # Round to the 3rd decimal place
    success_rate = round(successes / cfg.num_eval_episodes, 3)
    print(f'Success rate: {success_rate}')

def process_obs(obs, nets, device, image_keys, state_keys):
    # This function processes the observation such that they can just be fed into the model.
    # It should return a dictionary with the following keys
    # 'embed': The image embeddings
    # 'state': The state of the environment
    # You can change how you get this information depending on the environment.
    # print(obs.keys())


    state = [obs[state_key] for state_key in state_keys]
    # state.insert(3, np.sinh(obs['robot0_joint_pos_sin']))
    state = np.concatenate(state, axis=-1)

    with torch.no_grad():
        images = []
        for key in image_keys:
            img = torch.tensor(obs[key], dtype=torch.float32).permute(2, 0, 1).to(device)
            images.append(img)
        
        imgs = torch.stack(images).to(device)
        images = nets['vision_encoder'](imgs).cpu().flatten().numpy()
    return {
        'embed': images,
        'state': state
    }


def run_one_eval(env, nets: torch.nn.Module, config: DiffusionModelRunConfig, stats, noise_scheduler, device,
                 max_steps: int, render=True, save_path: Union[str, None]= None) -> bool:
    policy = DiffusionPolicy(config, nets, noise_scheduler, stats, device)
    # get first observation
    obs = env.reset()
    if save_path is not None:
        recorder = VideoRecorder()
        recorder.init(obs['agentview_image'])
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
            obs, reward, done, _ = env.step(action[i])

            if save_path is not None:
                recorder.record(obs['agentview_image'])

            if render:
                env.render()
            policy.add_obs(obs)
            # and reward/vis
            rewards.append(reward)

            # update progress bar
            step_idx += 1
            if step_idx > max_steps:
                recorder.save(save_path + "_fail.mp4")
                return False
            if reward > 0:
                recorder.save(save_path + "_success.mp4")
                return True

if __name__ == "__main__":
    main(EvalConfig())