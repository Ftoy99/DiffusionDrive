import bisect
from pathlib import Path
import logging
import os
import cv2
import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
import pytorch_lightning as pl
import torch
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import train, val
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from navsim.agents.hidden.hidden_config import HiddenConfig
from navsim.agents.hidden.hidden_features_nuscenes import HiddenFeatureBuilder, HiddenTargetBuilder, NuFeatureData, \
    NuTargetData

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"
# DATA_PATH = "/mnt/ds/nuscenes"
# VERSION = "v1.0-trainval"
DATA_PATH = "/mnt/ds/nuscenes_mini"
VERSION = "v1.0-mini"
front_cameras = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"]

def draw_bev(features,ego_fut_trajs_rel,path="/mnt/ds/debug/bev.png"):
    # features: torch.Tensor in [0,1], shape (C,H,W)
    bev_img_draw = features[0, :, :]  # (H,W)
    bev_img_draw = (bev_img_draw * 255).to(torch.uint8).cpu().numpy()  # <- convert to NumPy

    H, W = bev_img_draw.shape
    resolution = 0.1
    origin = np.array([W // 2, H // 2])

    for pos in ego_fut_trajs_rel:
        x_px = int(origin[0] - pos[1] / resolution)
        y_px = int(origin[1] - pos[0] / resolution)

        cv2.circle(bev_img_draw, (x_px, y_px), 2, 255, -1)

    cv2.imwrite(path, bev_img_draw)

def draw_semantic(bev_map_tensor, save_path="/mnt/ds/debug/bev_semantic.png"):
    """
    Draw BEV semantic map with distinct colors per class.
    :param bev_map_tensor: torch.Tensor (H x W)
    """
    bev_map = bev_map_tensor.cpu().numpy().astype(np.int32)
    # print(np.unique(bev_map))
    H, W = bev_map.shape

    # Assign distinct colors
    colors = {
        0: (0, 0, 0),  # background          → Black
        1: (128, 64, 128),  # road                → Dark Purple
        2: (0, 255, 0),  # walkway             → Green
        3: (255, 255, 0),  # centerline          → Yellow
        4: (192, 192, 192),  # static object       → Light Gray
        5: (0, 0, 255),  # vehicle             → Blue
        6: (255, 0, 0),  # pedestrian / human  → Red
    }

    # Create RGB image
    bev_img = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in colors.items():
        bev_img[bev_map == label] = color

    # Save image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bev_img)

def main():
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info("Loading scenes")
    nusc = NuScenes(version=VERSION, dataroot=DATA_PATH, verbose=True)
    logger.info("Loading CanBus")
    nusc_can_bus = NuScenesCanBus(DATA_PATH)
    logger.info(f"Loaded {len(train)} train scenes and {len(val)} valuation scenes")

    cfg = HiddenConfig()

    feature_builder = HiddenFeatureBuilder(cfg)
    target_builder = HiddenTargetBuilder(cfg)

    # TODO must clear this for each sample
    feat_data = NuFeatureData()
    target_data = NuTargetData()



    total_samples = sum([scene["nbr_samples"] for scene in nusc.scene])
    with tqdm(total=total_samples, desc="Processing NuScenes", unit="sample") as pbar:
        # For each scene
        for scene in nusc.scene:
            sample_token = scene["first_sample_token"]
            while sample_token != "":
                sample = nusc.get("sample", sample_token)

                #############################
                ## FEATURE DATA GENERATION ##
                #############################

                ego_fut_heading, ego_fut_trajs_rel , features = feature_data_preperation(feat_data, feature_builder,
                                                                                          nusc, nusc_can_bus, sample,
                                                                                          scene)

                #############################
                ##  TARGET DATA GENERATION ##
                #############################

                targets = target_data_preperation(ego_fut_heading, ego_fut_trajs_rel, nusc, sample, scene,
                                        target_builder, target_data)

                features = features
                targets = targets

                # Move to next sample
                exit()
                sample_token = sample["next"]
                pbar.update(1)


def target_data_preperation(ego_fut_heading, ego_fut_trajs_rel, nusc, sample, scene, target_builder,
                            target_data):
    # Future trajectories
    num_poses = min(5, len(ego_fut_heading))
    target_data.trajectory = [
        StateSE2(ego_fut_trajs_rel[i][0], ego_fut_trajs_rel[i][1], ego_fut_heading[i])
        for i in range(1, num_poses)
    ]
    # Get annotations
    annotations = []
    for annotation_id in sample['anns']:
        annotation = nusc.get("sample_annotation", annotation_id)
        annotations.append(annotation)
    target_data.annotations = annotations
    target_data.ego_pose_global_cords = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])['translation']
    target_data.ego_pose_heading = \
    Quaternion(nusc.get('ego_pose', sample['data']['LIDAR_TOP'])['rotation']).yaw_pitch_roll[0]

    map = NuScenesMap(
        dataroot=DATA_PATH,
        map_name=nusc.get('log', scene['log_token'])['location']
    )
    target_data.map = map
    map_api = NuScenesMapExplorer(map)
    target_data.map_api = map_api
    target = target_builder.compute_targets(target_data, nusc, sample)
    draw_semantic(target["bev_semantic_map"], f"/mnt/ds/debug/{sample['token']}_bev_semantic.png")
    return target


def feature_data_preperation(feat_data, feature_builder, nusc, nusc_can_bus, sample, scene):
    # Camera features
    for cam in front_cameras:
        sensor_token = sample['data'][cam]
        sample_data = nusc.get('sample_data', sensor_token)
        image_path = Path(nusc.dataroot) / sample_data['filename']

        # print(image_path)
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
    # Lidar debug image original
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
    save_path = Path(f"/mnt/ds/debug/{sample['token']}_lidar_bev.png")
    cv2.imwrite(str(save_path), bev_img)
    # print(f"Saved LiDAR BEV to {save_path}")
    # Ego status
    pose_data = nusc_can_bus.get_messages(scene["name"], "pose")
    # Multiple timestamps for each scene we need to filter with utime
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
        msg = before if abs(before['utime'] - sample_data_timestamp) < abs(
            after['utime'] - sample_data_timestamp) else after
    # get vel
    feat_data.ego_velocity = np.linalg.norm(np.array(msg['vel']))
    feat_data.ego_acceleration = np.linalg.norm(np.array(msg['accel']))
    # Driving command meta-action etc.
    # We are at sample T
    # We want the current position and then positions for x T+{1....x}
    # After we have the coordinates we want to translate them to our relative lidar So we have values close to 0.0 of current position
    # Based on the future waypoint poses/locations offsets we can determine the meta actions we must take.
    # if we are at a sample we dont have enough future trajectories keep the last meta-action
    num_of_samples_to_look = 7
    sample_cur = sample  # starting sample at time T
    ego_fut_trajs = {}
    ego_fut_heading = {}
    for i in range(num_of_samples_to_look + 1):
        # Get ego global position for this sample
        ego_pose = nusc.get('ego_pose', sample_cur['data']['LIDAR_TOP'])
        ego_fut_trajs[i] = ego_pose['translation']  # extract x, y, z
        ego_fut_heading[i] = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
        # Move to next sample if exists
        if sample_cur['next'] == '':
            break
        sample_cur = nusc.get('sample', sample_cur['next'])
    # Reference pose (current sample)
    ref_ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    ref_translation = np.array(ref_ego_pose['translation'])
    ref_rotation = Quaternion(ref_ego_pose['rotation']).inverse.rotation_matrix
    ego_fut_trajs_rel = []
    for i in range(len(ego_fut_trajs)):
        # Convert to numpy array
        pos = np.array(ego_fut_trajs[i])
        # Translate: move origin to current LiDAR
        pos_rel = pos - ref_translation
        # Rotate: align axes with current LiDAR
        pos_rel = ref_rotation @ pos_rel
        ego_fut_trajs_rel.append(pos_rel)
    ego_fut_trajs_rel = np.array(ego_fut_trajs_rel)
    # print(ego_fut_trajs_rel)
    ##ADD HERE
    last = ego_fut_trajs_rel[len(ego_fut_trajs_rel) - 1]  # final step
    if last[1] >= 2:
        command = np.array([1, 0, 0])  # Turn Right
        # print("Turn right")
    elif last[1] <= -2:
        command = np.array([0, 1, 0])  # Turn Left
        # print("Turn left")
    else:
        command = np.array([0, 0, 1])  # Go Straight
        # print("Go straight")
    feat_data.ego_driving_command = command
    feat_data.token = sample_cur["token"]
    features = feature_builder.compute_features(feat_data)
    draw_bev(features["lidar_feature"],ego_fut_trajs_rel,f"/mnt/ds/debug/{sample_cur['token']}_lidar_bev_img_with_traj_relative.png")
    return ego_fut_heading, ego_fut_trajs_rel, features


if __name__ == "__main__":
    main()
