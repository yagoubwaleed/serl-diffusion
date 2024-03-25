import collections
import json
import pickle

from itertools import chain

import h5py
import robosuite as suite
from dataclasses import dataclass
import tqdm.auto as tqdm
from diffusion_policy.configs import DatasetConfig


@dataclass
class DataReplayConfig:
    render: bool = False
    data_config: DatasetConfig = DatasetConfig()
    sim_json_path: str = "./data/square_peg.json"
    dataset_path: str = "./data/image.hdf5"
    output_path: str = "./data/peg_data.pkl"


def main(cfg: DataReplayConfig):
    metadata = json.load(open(cfg.sim_json_path, "r"))
    kwargs = metadata["env_kwargs"]
    if cfg.render:
        kwargs["has_renderer"] = True
    env = suite.make(
        env_name=metadata["env_name"],
        **kwargs
    )
    data = h5py.File(cfg.dataset_path, 'r')['data']
    trajectory_data = []
    for traj_key in tqdm.tqdm(data.keys()):
        traj = data[traj_key]
        extract_data_from_trajectory(traj, env, cfg)
        trajectory_data.append(extract_data_from_trajectory(traj, env, cfg))
    pickle.dump(trajectory_data, open(cfg.output_path, 'wb'))

def extract_data_from_trajectory(traj, env, cfg):
    results = collections.defaultdict(list)
    env.reset()
    env.sim.set_state_from_flattened(traj['states'][0])
    env.sim.forward()
    obs = env._get_observations()
    for i in range(len(traj['actions'])):

        action = traj['actions'][i]
        # Store the data:
        results['action'].append(action)
        # print(obs.keys())
        for key in chain(cfg.data_config.image_keys, cfg.data_config.state_keys):
            results[key].append(obs[key])
        obs, reward, done, _ = env.step(action)
        if cfg.render:
            env.render()
    return dict(results)


if __name__ == "__main__":
    main(DataReplayConfig())
