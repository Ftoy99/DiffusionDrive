import os

import cv2
import numpy as np
import pyquaternion
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from tqdm import tqdm

DATA_PATH = "/mnt/ds/nuscenes_mini"
VERSION = "v1.0-mini"

def draw_semantic(bev_map_tensor, save_path="/mnt/ds/debug/original/bev_semantic_debug.png"):
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


def get_ego_pose(sample, nusc):
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    # global location
    translation = ego_pose['translation']  # [x, y, z] in meters
    rotation = ego_pose['rotation']  # quaternion [w, x, y, z]

    return translation, rotation

def yaw_from_quaternion(q):
    quat = pyquaternion.Quaternion(q)
    return quat.yaw_pitch_roll[0]

def create_bev_semantic(sample,nusc : NuScenes,nusc_can_bus:NuScenesCanBus):
    semantic_map = None
    translation,rotation = get_ego_pose(sample,nusc)

    return semantic_map


if __name__ == '__main__':

    nusc = NuScenes(version=VERSION, dataroot=DATA_PATH, verbose=True)
    nusc_can_bus = NuScenesCanBus(DATA_PATH)

    total_samples = sum([scene["nbr_samples"] for scene in nusc.scene])
    with tqdm(total=total_samples, desc="Processing NuScenes", unit="sample") as pbar:
        # For each scene
        for scene in nusc.scene:
            sample_token = scene["first_sample_token"]
            while sample_token != "":
                sample = nusc.get("sample", sample_token)

                bev_sematic = create_bev_semantic(sample,nusc,nusc_can_bus)
                draw_semantic(bev_sematic)
                sample_token = sample["next"]
                pbar.update(1)