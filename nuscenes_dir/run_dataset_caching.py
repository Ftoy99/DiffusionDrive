import bisect
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import uuid
import os

import cv2
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_utils import worker_map
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import train, val
from nuscenes.utils.data_classes import LidarPointCloud
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
    nusc_can_bus = NuScenesCanBus(DATA_PATH)
    logger.info(f"Loaded {len(train)} train scenes and {len(val)} valuation scenes")

    cfg = HiddenConfig()

    feature_builder = HiddenFeatureBuilder(cfg)
    target_builder = HiddenTargetBuilder(cfg)

    # TODO must clear this for each sample
    feat_data = NuFeatureData()
    target_data = NuTargetData()

    #For each scene
    for scene in nusc.scene:
        first_sample = scene["first_sample_token"]
        sample = nusc.get('sample', first_sample)

        # Camera features
        for cam in front_cameras:
            sensor_token = sample['data'][cam]
            sample_data = nusc.get('sample_data', sensor_token)
            image_path = Path(nusc.dataroot) / sample_data['filename']

            print(image_path)

            # Load image as NumPy array in RGB
            img = cv2.imread(str(image_path))[:, :, ::-1]  # BGR → RGB
            feat_data.images[cam] = img  # shape (H, W, 3), dtype=uint8

        # LiDAR features
        lidar_token = sample['data']['LIDAR_TOP']
        sample_data = nusc.get('sample_data', lidar_token)
        sample_data_timestamp = sample_data['timestamp']
        lidar_path = Path(nusc.dataroot) / sample_data['filename']

        # Load point cloud
        pc = LidarPointCloud.from_file(str(lidar_path))
        points = pc.points[:3, :].T  # shape (N, 3) -> x,y,z

        feat_data.lidar = points  # store in your feature data


        #Lidar debug image original
        lidar_points = feat_data.lidar  # (N, 3)
        bev_size = (512, 512)  # output image size
        x_min, x_max = -50, 50
        y_min, y_max = -50, 50
        x_img = ((lidar_points[:, 0] - x_min) / (x_max - x_min) * (bev_size[0] - 1)).astype(np.int32)
        y_img = ((lidar_points[:, 1] - y_min) / (y_max - y_min) * (bev_size[1] - 1)).astype(np.int32)
        x_img = np.clip(x_img, 0, bev_size[0] - 1)
        y_img = np.clip(y_img, 0, bev_size[1] - 1)
        bev_img = np.zeros(bev_size, dtype=np.uint8)
        bev_img[y_img, x_img] = 255

        save_path = Path("/mnt/ds/debug/lidar_bev.png")
        cv2.imwrite(str(save_path), bev_img)
        print(f"Saved LiDAR BEV to {save_path}")

        #Ego status
        pose_data = nusc_can_bus.get_messages(scene["name"],"pose")
        #Multiple timestamps for each scene we need to filter with utime
        # extract timestamps
        times = [msg['utime'] for msg in pose_data]

        # find closest CAN timestamp
        idx = bisect.bisect_left(times, sample_data_timestamp)
        if idx == 0:
            msg = pose_data[0]
        elif idx == len(times):
            msg = pose_data[-1]
        else:
            before = pose_data[idx - 1]
            after = pose_data[idx]
            msg = before if abs(before['utime'] - sample_data_timestamp) < abs(after['utime'] - sample_data_timestamp) else after

        # get vel
        feat_data.ego_velocity = np.linalg.norm(np.array(msg['vel']))
        feat_data.ego_acceleration = np.linalg.norm(np.array(msg['accel']))


        # Driving command meta-action etc.
        # We are at sample T
        # We want the current position and then positions for x T+{1....x}
        # After we have the coordinates we want to translate them to our relative lidar So we have values close to 0.0 of current position

        #Based on the future waypoint poses/locations offsets we can determine the meta actions we must take.


        #if we are at a sample we dont have enough future trajectories keep the last meta-action
        num_of_samples_to_look = 7
        sample_cur = sample  # starting sample at time T
        ego_fut_trajs = {}
        for i in range(num_of_samples_to_look + 1):
            # Get ego global position for this sample
            ego_pose = nusc.get('ego_pose',sample_cur['data']['LIDAR_TOP'])
            ego_fut_trajs[i] = ego_pose['translation']  # extract x, y, z

            # Move to next sample if exists
            if sample_cur['next'] == '':
                break
            sample_cur = nusc.get('sample', sample_cur['next'])

        print(ego)

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
