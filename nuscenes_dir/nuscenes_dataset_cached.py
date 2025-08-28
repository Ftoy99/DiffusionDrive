import gzip
import os
import pickle

import torch
from torch.utils.data import Dataset


class SimpleCacheDataset(Dataset):
    """Dataset for multiple features.gz / target.gz per split folder."""

    def __init__(self, cache_path: str, split: str):
        split_path = os.path.join(cache_path, split)
        self.features = []
        self.targets = []
        for folder in os.listdir(split_path):
            folder_path = os.path.join(split_path, folder)
            if not os.path.isdir(folder_path):
                continue

            with gzip.open(os.path.join(folder_path, "features.gz"), "rb") as f:
                self.features.append(pickle.load(f))
            with gzip.open(os.path.join(folder_path, "target.gz"), "rb") as f:
                self.targets.append(pickle.load(f))

        assert len(self.features) == len(self.targets), "Features/targets mismatch"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y


def dict_collate(batch):
    """
    batch: list of tuples (x_dict, y_dict)
    Output: dicts of tensors with shape Batch , _____
    """
    batch_x, batch_y = zip(*batch)

    collated_x = {key: torch.stack([d[key] for d in batch_x], dim=0) for key in batch_x[0]}
    collated_y = {key: torch.stack([d[key] for d in batch_y], dim=0) for key in batch_y[0]}

    return collated_x, collated_y
