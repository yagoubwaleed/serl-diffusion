import h5py
import numpy as np
import torch
import os
import pickle
import cv2

from diffusion_policy.configs import DatasetConfig


def create_sample_indices(
        episode_ends: np.ndarray, sequence_length: int,
        pad_before: int = 0, pad_after: int = 0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        '''
        This is the base class for all datasets. You have to populate the following fields:
        - indices: a numpy array of shape (N, 4) where N is the number of samples. Each row is a tuple of
        (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        - stats: a dictionary of statistics for each data type
        - normalized_train_data: a dictionary of normalized data
        - pred_horizon: the prediction horizon
        - action_horizon: the action horizon
        - obs_horizon: the observation horizon
        '''
        self.indices = None
        self.stats = None
        self.normalized_train_data = None
        self.pred_horizon = None
        self.action_horizon = None
        self.obs_horizon = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon, :]
        nsample['state'] = nsample['state'][:self.obs_horizon, :]
        return nsample


class JacobPickleDataset(BaseDataset):
    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int, action_horizon: int,
                 num_trajectories: int, image_keys, state_keys, action_key="action"):
        data = pickle.load(open(dataset_path, 'rb'))

        actions = []
        images = []
        states = []
        episode_ends = []

        for trajectory in data:
            if num_trajectories > 0:
                num_trajectories -= 1
                if num_trajectories == 0:
                    break

            state = [np.array(trajectory[state_key]) for state_key in state_keys]
            state = np.concatenate(state, axis=-1)
            states.append(state)
            imgs = [np.array(trajectory[key]) for key in image_keys]
            for i in range(len(imgs)):
                # resize image to (T, 96, 96, C)
                imgs[i] = np.array([cv2.resize(img, (96, 96)) for img in imgs[i]])

                # (T, H, W, C) -> (T, C, H, W)
                imgs[i] = np.moveaxis(imgs[i], 3, 1)
            # (T, N, C, H, W)
            imgs = np.stack(imgs, axis=1) if len(imgs) else np.empty((len(state), 0, 1, 96, 96))
            images.append(imgs)
            actions.append(np.array(trajectory[action_key]))
            if len(episode_ends) == 0:
                episode_ends.append(len(state))
            else:
                episode_ends.append(episode_ends[-1] + len(state))
        del data
        actions = np.concatenate(actions).astype(np.float32)
        states = np.concatenate(states).astype(np.float32)
        episode_ends = np.array(episode_ends)
        images = np.concatenate(images).astype(np.float32)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'state': states,
            'action': actions,
        }

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = images

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon


class D4RLDataset(JacobPickleDataset):
    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int, action_horizon: int,
                 num_trajectories: int, image_keys):
        super().__init__(dataset_path, pred_horizon, obs_horizon, action_horizon, num_trajectories, image_keys, ["observations"], "actions")

class HD5PYDataset(BaseDataset):
    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int, action_horizon: int,
                 num_trajectories: int = 200, image_keys: list = None,
                 state_keys: list = None):

        # load the demonstration dataset:
        super().__init__()
        assert state_keys is not None and image_keys is not None

        data = h5py.File(dataset_path, 'r')['data']

        actions = []
        images = []
        states = []
        episode_ends = []

        for traj_key in data.keys():
            if num_trajectories > 0:
                num_trajectories -= 1
                if num_trajectories == 0:
                    break
            trajectory = data[traj_key]

            state = [np.array(trajectory[f'obs/{state_key}']) for state_key in state_keys]
            state = np.concatenate(state, axis=-1)
            states.append(state)
            imgs = [np.array(trajectory[f'obs/{key}']) for key in image_keys]
            for i in range(len(imgs)):
                # resize image to (T, 96, 96, C)
                imgs[i] = np.array([cv2.resize(img, (96, 96)) for img in imgs[i]])

                # (T, H, W, C) -> (T, C, H, W)
                imgs[i] = np.moveaxis(imgs[i], 3, 1)
            imgs = np.stack(imgs, axis=1)
            images.append(imgs)
            actions.append(np.array(trajectory['actions']))
            if len(episode_ends) == 0:
                episode_ends.append(len(state))
            else:
                episode_ends.append(episode_ends[-1] + len(state))
        del data
        actions = np.concatenate(actions).astype(np.float32)
        states = np.concatenate(states).astype(np.float32)
        episode_ends = np.array(episode_ends)
        images = np.concatenate(images).astype(np.float32)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'state': states,
            'action': actions,
        }

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = images

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

########## modified to flatten hil serl dictionary (dastaset ccompatibility) ##########
class SERLImageDataset(BaseDataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 num_trajectories: int = 100,
                 image_keys: list = ['wrist_1', 'wrist_2'],
                 state_keys: list = ['state'],
                 ):

        # load the demonstration dataset:
        data = pickle.load(open(dataset_path, 'rb'))

        actions = []
        images = []
        states = []
        episode_ends = []

        for i in range(len(data)):
            # ERROR CODE: state = [np.array(data[i]["observations"]["state"][:, state_key]) for state_key in state_keys]
            # breakpoint()
            state = [np.array(data[i]["observations"][state_key]) for state_key in state_keys]
            state = np.concatenate(state, axis=-1).flatten()
            states.append(state)
            imgs = []
            for key in image_keys:
                img = data[i]['observations'][key]
                if img.ndim == 4:
                    img = img[0]
                img = np.moveaxis(img, -1, 0)
                imgs.append(img)
            # breakpoint()
            image = np.stack(imgs, axis=0)
            images.append(image)
            actions.append(data[i]['actions'])
            if data[i]['dones']:
                episode_ends.append(i + 1)
                # Temp solution
                if len(episode_ends) >= num_trajectories and num_trajectories != -1:
                    break

        actions = np.array(actions).astype(np.float32)
        states = np.array(states).astype(np.float32)
        episode_ends = np.array(episode_ends)
        images = np.array(images).astype(np.float32)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'state': states,
            'action': actions,
        }

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = images

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon


if __name__ == "__main__":
    dataset_path = '/home/robot/projects/waleed-test/jax-hitl-hil-serl/examples/experiments/cube_reach3/demos/cube_reach3_200_success_images_2025-04-11_16-01-07.pkl'
    print('here')
    try:
        cfg = DatasetConfig()
        dataset1 = SERLImageDataset(
            dataset_path=dataset_path,
            pred_horizon=16,
            obs_horizon=2,
            action_horizon=8,
            num_trajectories=-1,
            image_keys=cfg.image_keys,
            state_keys=cfg.state_keys,
        )
    except Exception as e:
        print(e)
    print('here')
    print(len(dataset1))
    # breakpoint()
