import os
import gzip
import pickle
import logging

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.hidden.hidden_agent import HiddenAgent
from navsim.agents.hidden.hidden_config import HiddenConfig

from navsim.planning.training.dataset import Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CACHE_PATH = "/mnt/ds/nuscenes_cached_mini"
OUTPUT_DIR = "/mnt/ds/debug/log"
class SimpleCacheDataset(Dataset):
    """Dataset for multiple features.gz / target.gz per split folder."""

    def __init__(self, cache_path: str, split: str):
        split_path = os.path.join(cache_path, split)
        self.features, self.targets = [], []

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
    """
    batch_x, batch_y = zip(*batch)

    # Collate dicts: keep each key as a list of tensors (variable-size)
    collated_x = {key: [d[key] for d in batch_x] for key in batch_x[0]}
    collated_y = {key: [d[key] for d in batch_y] for key in batch_y[0]}

    return collated_x, collated_y


def main() -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(0, workers=True)
    logger.info(f"Global Seed set to {0}")

    logger.info(f"Path where all results are stored: {OUTPUT_DIR}")

    logger.info("Building Agent")
    cfg = HiddenConfig()
    agent: AbstractAgent = HiddenAgent(cfg,lr=1e-4,checkpoint_path=None)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    train_data = SimpleCacheDataset(
        cache_path=CACHE_PATH,
        split="train"
    )
    val_data = SimpleCacheDataset(
        cache_path=CACHE_PATH,
        split="val"
    )

    def dict_collate(batch):
        """
        batch: list of tuples (x_dict, y_dict)
        Output: dicts of tensors with shape Batch , _____
        """
        batch_x, batch_y = zip(*batch)

        collated_x = {key: torch.stack([d[key] for d in batch_x], dim=0) for key in batch_x[0]}
        collated_y = {key: torch.stack([d[key] for d in batch_y], dim=0) for key in batch_y[0]}

        return collated_x, collated_y

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, batch_size=64,prefetch_factor=2,num_workers=4,collate_fn=dict_collate, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))

    val_dataloader = DataLoader(val_data,  batch_size=64,prefetch_factor=2,num_workers=4,collate_fn=dict_collate, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(max_epochs=10,check_val_every_n_epoch=1,val_check_interval=1,limit_val_batches=1,
                         limit_train_batches=1,accelerator="gpu",strategy="ddp_find_unused_parameters_true",
                         precision="16-mixed",num_nodes=1,num_sanity_val_steps=0,fast_dev_run=False,
                         accumulate_grad_batches=1,gradient_clip_val=1.0,gradient_clip_algorithm="norm",
                         default_root_dir=OUTPUT_DIR,callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
