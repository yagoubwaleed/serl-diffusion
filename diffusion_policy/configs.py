from dataclasses import dataclass, field
from typing import Dict

import hydra
from hydra.conf import HydraConf
from omegaconf import OmegaConf

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
class DatasetConfig:
    type: str = "HDF5"  # Options are SERL or HDF5
    dataset_path: str = "${hydra:runtime.cwd}/data/two_camera_views.hdf5"
    num_traj: int = -1
    image_keys: list = field(default_factory=lambda: ['agentview_image', 'robot0_eye_in_hand_image'])
    state_keys: list = field(default_factory=lambda:['robot0_proprio-state'])

@dataclass
class DiffusionModelRunConfig:
    hydra: ExperimentHydraConfig = ExperimentHydraConfig()
    dataset: DatasetConfig = DatasetConfig()
    device: str = "cuda"
    checkpoint_path: str = "${hydra:runtime.cwd}/image_lift_propreo.pt"

    batch_size: int = 256//2
    num_epochs: int = 8
    with_state: bool = True
    state_len: int = 45
    action_dim: int = 7
    pred_horizon: int = 12
    obs_horizon: int = 4
    action_horizon: int = 8
    num_diffusion_iters: int = 100
    num_cameras: int = 2





