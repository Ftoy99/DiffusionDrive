import gzip
import os
import pickle

import torch
from torch.utils.data import Dataset


class SimpleCacheDataset(Dataset):
    def __init__(self, cache_path: str, split: str):
        split_path = os.path.join(cache_path, split)
        self.samples = []
        for folder in os.listdir(split_path):
            folder_path = os.path.join(split_path, folder)
            if not os.path.isdir(folder_path):
                continue
            self.samples.append({
                "features": os.path.join(folder_path, "features.gz"),
                "target": os.path.join(folder_path, "target.gz"),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with gzip.open(sample["features"], "rb") as f:
            x = pickle.load(f)
        with gzip.open(sample["target"], "rb") as f:
            y = pickle.load(f)
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
