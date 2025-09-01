import logging

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.hidden.hidden_agent import HiddenAgent
from navsim.agents.hidden.hidden_config import HiddenConfig
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from nuscenes_dir.nuscenes_dataset_cached import SimpleCacheDataset, dict_collate

logger = logging.getLogger(__name__)

CACHE_PATH = "/mnt/ds/nuscenes_cached"
OUTPUT_DIR = "/mnt/ds/debug/log"
CHECKPOINT_PATH = "/mnt/ds/debug/log/lightning_logs/version_28/checkpoints/e9.ckpt"

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
    agent: AbstractAgent = HiddenAgent(cfg, lr=1e-4, checkpoint_path=CHECKPOINT_PATH)

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


    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, batch_size=64, prefetch_factor=2, num_workers=4, collate_fn=dict_collate,
                                  shuffle=True)
    logger.info("Num training samples: %d", len(train_data))

    val_dataloader = DataLoader(val_data, batch_size=64, prefetch_factor=2, num_workers=4, collate_fn=dict_collate,
                                shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(max_epochs=20, check_val_every_n_epoch=1, val_check_interval=1.0, limit_val_batches=1.0,
                         limit_train_batches=1.0, accelerator="gpu", strategy="ddp_find_unused_parameters_true",
                         precision="16-mixed", num_nodes=1, num_sanity_val_steps=0, fast_dev_run=False,
                         accumulate_grad_batches=1, gradient_clip_val=1.0, gradient_clip_algorithm="norm",
                         default_root_dir=OUTPUT_DIR, callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
