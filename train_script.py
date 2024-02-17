import os
import time
from typing import List
import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from diffusion_policy.dataset import SERLImageDataset, normalize_data, unnormalize_data
from diffusion_policy.networks import ConditionalUnet1D, get_resnet, replace_bn_with_gn

import hydra
from dataclasses import dataclass
from diffusion_policy.hydra_utils import ExperimentHydraConfig


### ======================= Hydra Configuration  =============================


@dataclass
class ImageTrainingScriptConfig:
    hydra: ExperimentHydraConfig = ExperimentHydraConfig()
    num_trajs: int = 50
    batch_size: int = 64
    num_epochs: int = 10
    checkpoint_path: str = "${hydra:runtime.cwd}/${checkpoint_name: ${num_trajs}}"
    dataset_path: str = "${hydra:runtime.cwd}/peg_insert_100_demos_2024-02-11_13-35-54.pkl"
    with_state: bool = True
    state_len: int = 19
    action_dim: int = 6

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="image_train_script", node=ImageTrainingScriptConfig)


@hydra.main(version_base=None, config_name="image_train_script")
def main(cfg: ImageTrainingScriptConfig):
    load_pretrained = True
    num_epochs = cfg.num_epochs
    ckpt_name = cfg.checkpoint_path
    with_state = cfg.with_state
    state_len = cfg.state_len
    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    num_diffusion_iters = 100
    num_cameras = 2
    action_dim = cfg.action_dim


    # create dataset from file
    dataset = SERLImageDataset(
        dataset_path=cfg.dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        num_trajectories=cfg.num_trajs,
    )

    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    vision_encoder = get_resnet('resnet18')
    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    vision_feature_dim = 512 * num_cameras
    # ResNet18 has output dim of 512
    obs_dim = vision_feature_dim + (state_len if with_state else 0)


    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parameter are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    if load_pretrained:
        state_dict = torch.load(cfg.checkpoint_path, map_location='cuda')
        ema_nets = nets
        ema_nets.load_state_dict(state_dict)
        print('Pretrained weights loaded.')
    else:
        print('Training from scratch.')
        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        nimage = nbatch['image'][:, :obs_horizon].to(device)
                        naction = nbatch['action'].to(device)
                        B = nimage.shape[0]


                        image_features = nets['vision_encoder'](
                            nimage.flatten(end_dim=2))
                        # print(nimage.flatten(end_dim=2).shape)
                        # time.sleep(100)
                        image_features = image_features.reshape(
                            B, obs_horizon, vision_feature_dim)
                        # (B,obs_horizon,D)
                        # concatenate vision feature and low-dim obs
                        obs_features = image_features
                        if with_state:
                            nagent_pos = nbatch['state'][:, :obs_horizon].to(device)
                            obs_features = torch.cat([image_features, nagent_pos], dim=-1)

                        obs_cond = obs_features.flatten(start_dim=1)
                        # (B, obs_horizon * obs_dim)

                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=device)

                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (B,), device=device
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = noise_scheduler.add_noise(
                            naction, noise, timesteps)
                        # predict the noise residual
                        noise_pred = noise_pred_net(
                            noisy_actions, timesteps, global_cond=obs_cond)

                        # L2 loss
                        loss = nn.functional.mse_loss(noise_pred, noise)
                        time.sleep(2)

                        # optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        ema.step(nets.parameters())

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))

        # Weights of the EMA model
        # is used for inference
        ema_nets = nets
        ema.copy_to(ema_nets.parameters())
        # save checkpoint
        torch.save(ema_nets.state_dict(), cfg.checkpoint_path)
        print('done')

    if not load_pretrained:
        return

    # limit enviornment interaction to 200 steps before termination
    max_steps = 200
    # env = PushTEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    # env.seed(100000)
    import gymnasium as gym
    from franka_env.envs.wrappers import (
        GripperCloseEnv,
        SpacemouseIntervention,
        Quat2EulerWrapper,
        SERLObsWrapper
    )

    env = gym.make(
        "FrankaPegInsert-Vision-v0",
        fake_env=False,

    )
    env = GripperCloseEnv(env)
    # env = SpacemouseIntervention(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)

    def process_obs(obs):
        state = obs['state'] #[4:10]
        with torch.no_grad():
            im1 = torch.tensor(obs['wrist_1'], dtype=torch.float32).permute(2, 0, 1).to(device)
            im2 = torch.tensor(obs['wrist_2'], dtype=torch.float32).permute(2, 0, 1).to(device)
            im_stack = torch.stack([im1, im2], dim=0)
            images = ema_nets['vision_encoder'](im_stack).cpu().flatten().numpy()
        return {
            'embed': images,
            'state': state
        }

    def run_one_eval():
        # get first observation
        obs, info = env.reset()
        obs = process_obs(obs)

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x['embed'] for x in obs_deque])
                if with_state:
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
                    image_features = nimages  # ema_nets['vision_encoder'](nimages)
                    # (2,1024)

                    # concat with low-dim observations
                    if with_state:
                        obs_features = torch.cat([image_features, nagent_poses], dim=-1)
                    else:
                        obs_features = image_features

                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, pred_horizon, action_dim), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = ema_nets['noise_pred_net'](
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
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end, :]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs = process_obs(obs)
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        return False
                    if done:
                        return True

    successes = 0
    for i in range(10):
        result = run_one_eval()
        successes += result

    # print out the maximum target coverage
    # print('Score: ', max(rewards))

    print('Successes: ', successes)


if __name__ == "__main__":
    main()
