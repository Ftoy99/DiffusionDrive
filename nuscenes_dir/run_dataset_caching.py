from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import uuid
import os

import cv2
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_utils import worker_map
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
from nuscenes import NuScenes
from nuscenes.utils.splits import train, val

from navsim.agents.hidden.hidden_config import HiddenConfig
from navsim.agents.hidden.hidden_features_nuscenes import HiddenFeatureBuilder, HiddenTargetBuilder, NuFeatureData, \
    NuTargetData
from navsim.planning.training.dataset import Dataset
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"
# DATA_PATH = "/mnt/ds/nuscenes"
# VERSION = "v1.0-trainval"
DATA_PATH = "/mnt/ds/nuscenes_mini"
VERSION = "v1.0-mini"
front_cameras = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"]


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    Helper function to cache features and targets of learnable agent.
    :param args: arguments for caching
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    agent: AbstractAgent = instantiate(cfg.agent)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    logger.info(f"Extracted {len(scene_loader.tokens)} scenarios for thread_id={thread_id}, node_id={node_id}.")

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    return []


# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main():
    """
    Main entrypoint for dataset caching script.
    :param cfg: omegaconf dictionary
    """
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    # logger.info("Building Worker")
    # worker = RayDistributed(
    #     master_node_ip=None,
    #     threads_per_node=None,  # use all available threads
    #     debug_mode=False,
    #     log_to_driver=True,
    #     logs_subdir="logs",
    #     use_distributed=False,  # single-PC mode
    # )

    logger.info("Loading scenes")
    nusc = NuScenes(version=VERSION, dataroot=DATA_PATH, verbose=True)

    logger.info(f"Loaded {len(train)} train scenes and {len(val)} valuation scenes")

    cfg = HiddenConfig()

    feature_builder = HiddenFeatureBuilder(cfg)
    target_builder = HiddenTargetBuilder(cfg)

    feat_data = NuFeatureData()
    target_data = NuTargetData()

    for scene in nusc.scene:
        first_sample = scene["first_sample_token"]
        sample = nusc.get('sample', first_sample)

        for cam in front_cameras:
            sensor_token = sample['data'][cam]
            sample_data = nusc.get('sample_data', sensor_token)
            image_path = Path(nusc.dataroot) / sample_data['filename']

            print(image_path)

            # Load image as NumPy array in RGB
            img = cv2.imread(str(image_path))[:, :, ::-1]  # BGR → RGB
            feat_data.images[cam] = img  # shape (H, W, 3), dtype=uint8

        features = feature_builder.compute_features(feat_data)
        target = target_builder.compute_targets(target_data)

    # data_points = [
    #     {
    #         "cfg": cfg,
    #         "log_file": log_file,
    #         "tokens": tokens_list,
    #     }
    #     for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    # ]
    #
    # _ = worker_map(worker, cache_features, data_points)
    # logger.info(f"Finished caching {len(scene_loader)} scenarios for training/validation dataset")


if __name__ == "__main__":
    main()
