import hydra
from diffusers import DDPMScheduler, EMAModel, get_scheduler
from hydra.conf import HydraConf
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.dataset import SERLImageDataset
from diffusion_policy.networks import get_resnet, replace_bn_with_gn, ConditionalUnet1D

OmegaConf.register_new_resolver("slash_to_dot", lambda dir: dir.replace("/", "."))
OmegaConf.register_new_resolver("checkpoint_name", lambda num_trajs: f"checkpoint_w_{num_trajs}_trajectories.pt")
OmegaConf.register_new_resolver("compute_epochs", lambda num_trajs: (110 - num_trajs) * 10)


@dataclass
class ExperimentHydraConfig(HydraConf):
    root_dir_name: str = "./outputs"
    new_override_dirname: str = "${slash_to_dot: ${hydra:job.override_dirname}}"
    run: Dict = field(default_factory=lambda: {
        # A more sophisticated example:
        # "dir": "${hydra:root_dir_name}/${hydra:new_override_dirname}/seed=${seed}/${now:%Y-%m-%d_%H-%M-%S}",
        # Default behavior logs by date and script name:
        "dir": "${hydra:root_dir_name}/${now:%Y-%m-%d_%H-%M-%S}",
    }
                      )

    sweep: Dict = field(default_factory=lambda: {
        "dir": "${..root_dir_name}/multirun/${now:%Y-%m-%d_%H-%M-%S}",
        "subdir": "${hydra:new_override_dirname}",
    }
                        )

    job: Dict = field(default_factory=lambda: {
        "config": {
            "override_dirname": {
                "exclude_keys": [
                    "sim_device",
                    "rl_device",
                    "headless",
                ]
            }
        },
        "chdir": True
    })


@dataclass
class ImageTrainingScriptConfig:
    hydra: ExperimentHydraConfig = ExperimentHydraConfig()
    device: str = "cuda"
    num_trajs: int = 10
    batch_size: int = 64
    num_epochs: int = 6
    checkpoint_path: str = "${hydra:runtime.cwd}/${checkpoint_name: ${num_trajs}}"
    dataset_path: str = "${hydra:runtime.cwd}/peg_insert_100_demos_2024-02-11_13-35-54.pkl"
    with_state: bool = True
    state_len: int = 19
    action_dim: int = 6
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    num_diffusion_iters: int = 100
    num_cameras: int = 2



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="image_train_script", node=ImageTrainingScriptConfig)


def instantiate_model(cfg: ImageTrainingScriptConfig, model_only: bool = False):
    '''
    Instantiate the model and the training objects.
    If model only, returns network and scheduler and device only
    If not model only, returns network, ema, noise scheduler, optimizer, lr_scheduler, dataloader, stats, device
    '''
    device = torch.device('cuda')

    vision_feature_dim = 512 * cfg.num_cameras
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # ResNet18 has output dim of 512
    obs_dim = vision_feature_dim + (cfg.state_len if cfg.with_state else 0)

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=cfg.action_dim,
        global_cond_dim=obs_dim * cfg.obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    # device transfer
    _ = nets.to(device)

    if model_only:
        return nets, noise_scheduler, device
    # create dataset from file
    dataset = SERLImageDataset(
        dataset_path=cfg.dataset_path,
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        num_trajectories=cfg.num_trajs,
    )
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
        num_warmup_steps=(len(dataloader) * cfg.num_epochs) // 10,
        num_training_steps=len(dataloader) * cfg.num_epochs
    )
    return nets, ema, noise_scheduler, optimizer, lr_scheduler, dataloader, stats, device
