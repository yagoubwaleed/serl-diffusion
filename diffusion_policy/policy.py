from typing import Any, Dict
import collections

import numpy as np
import torch

from diffusers import DDPMScheduler
from diffusion_policy.configs import DiffusionModelRunConfig
from diffusion_policy.make_networks import instantiate_model_artifacts
from diffusion_policy.dataset import normalize_data, unnormalize_data


@torch.no_grad()
def copy_obs(obs: Dict[Any, torch.Tensor]):
    return {k:v.clone() for k,v in obs.items()}


class DiffusionPolicy(object):
    def __init__(self, config: DiffusionModelRunConfig, nets: torch.ModuleDict, noise_scheduler: DDPMScheduler, stats, device):
        self.config = config
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.obs_deque = collections.deque([], maxlen=config.obs_horizon)
        self.stats = self._torchify_dict(stats)

    @staticmethod
    def load(chkpt_path: str):
        checkpoint = torch.load(chkpt_path, map_location='cuda')
        diff_run_config: DiffusionModelRunConfig = checkpoint['config']

        nets, noise_scheduler, device = instantiate_model_artifacts(diff_run_config, model_only=True)
        nets.load_state_dict(checkpoint['state_dict'])
        print('Pretrained weights loaded.')
        stats = checkpoint['stats']
        return DiffusionPolicy(diff_run_config, nets, noise_scheduler, stats, device)

    def _torchify_dict(self, d: dict) -> dict:
        ret = {}
        for k, v in d.items():
            if isinstance(v, (np.ndarray, list)):
                v = self._torchify(v)
            elif isinstance(v, dict):
                v = self._torchify_dict(v)
            ret[k] = v
        return ret

    def _torchify(self, arr) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _process_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        ret = {}
        if self.config.with_image:
            ims = []
            for k in self.config.dataset.image_keys:
                im = torch.as_tensor(obs[k], dtype=torch.float32).permute(2, 0, 1).to(self.device)
                ims.append(im)
            im_stack = torch.stack(ims, dim=0)
            images: torch.Tensor = self.nets['vision_encoder'](im_stack).flatten()
            ret["embed"] = images
        if self.config.with_state:
            ret["state"] = torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device)
        return ret

    @torch.no_grad()
    def get_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Takes dict with "embed" and "state" possibly as keys, and returns actions.
        actions is a (H, D) array, where H is the action horizon and D is action dimension.
        """
        o = self._process_obs(obs)
        self.obs_deque.append(o)
        while len(self.obs_deque) < self.config.obs_horizon:
            self.obs_deque.append(copy_obs(o))

        if self.config.with_image:
            nimages = torch.stack([x['embed'] for x in self.obs_deque])
        if self.config.with_state:
            states = torch.stack([x["state"] for x in self.obs_deque])
            states = normalize_data(states, stats=self.stats["state"])

        obs_features = []
        if self.config.with_image:
            obs_features.append(nimages)
        if self.config.with_state:
            obs_features.append(states)
        obs_features = torch.cat(obs_features, dim=-1)

        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        # initialize action from Gaussian noise
        noisy_action = torch.randn(
            (1, self.config.pred_horizon, self.config.action_dim), device=self.device)
        naction = noisy_action

        # init scheduler
        self.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.nets['noise_pred_net'](
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred: torch.Tensor = unnormalize_data(naction, stats=self.stats['action'])

        start = self.config.obs_horizon - 1
        end = start + self.config.action_horizon
        action = action_pred[start:end, :].cpu().numpy()

        return action

def main():
    import time
    from tqdm import tqdm
    policy = DiffusionPolicy.load("chkpt.pt")
    # policy.config.num_diffusion_iters = 100
    n_iter = 30
    times = []
    for _ in tqdm(range(n_iter)):
        state = np.random.rand(policy.config.state_len)
        obs = {"state": state}
        start = time.time()
        policy.get_action(obs)
        end = time.time()
        times.append(end-start)
    print(f"Avg Freq: {n_iter / np.sum(times):.3f}hz")
    print(f"Max Freq: {1 / np.min(times):.3f}hz")

if __name__ == "__main__":
    main()
