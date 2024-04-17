import collections
import json
import os
from typing import Callable, Union
import robosuite as suite
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm
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
    imgs = [np.array(obs[key]) for key in image_keys]

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
    # get first observation
    obs = env.reset()
    if save_path is not None:
        recorder = VideoRecorder()
        recorder.init(obs['agentview_image'])
    obs = process_obs(obs, nets, device, config.dataset.image_keys, config.dataset.state_keys)

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * config.obs_horizon, maxlen=config.obs_horizon)
    # save visualization and rewards
    rewards = list()
    done = False
    step_idx = 0


    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        images = np.stack([x['embed'] for x in obs_deque])
        if config.with_state:
            agent_poses = np.stack([x['state'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['state'])
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

        # images are already normalized to [0,1]
        nimages = images

        # device transfer
        nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
        # (2,3,96,96)
        # (2,2)

        # infer action
        with torch.no_grad():
            # get image features
            image_features = nimages
            # (2,1024)

            # concat with low-dim observations
            if config.with_state:
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)
            else:
                obs_features = image_features

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Gaussian noise
            noisy_action = torch.randn(
                (B, config.pred_horizon, config.action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(config.num_eval_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = config.obs_horizon - 1
        end = start + config.action_horizon
        action = action_pred[start:end, :]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _ = env.step(action[i])

            if save_path is not None:
                recorder.record(obs['agentview_image'])


            if render:
                env.render()
            obs = process_obs(obs, nets, device, config.dataset.image_keys, config.dataset.state_keys)
            obs_deque.append(obs)
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