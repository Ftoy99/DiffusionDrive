import math
import os

import cv2
import numpy as np
import pyquaternion
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from pyquaternion import Quaternion
from tqdm import tqdm
import torch

from navsim.agents.hidden.hidden_config import HiddenConfig

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

def create_bev_semantic(sample, nusc: NuScenes, nusc_can_bus: NuScenesCanBus):
    cfg = HiddenConfig()
    zoom = 4.0  # 2× zoom, increase to zoom in more
    # Make a large square canvas to avoid clipping after rotation
    max_dim = max(cfg.bev_semantic_frame)
    bev_canvas = np.zeros((max_dim, max_dim), dtype=np.uint8)
    translation, rotation = get_ego_pose(sample, nusc)

    # Get map name from sample
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_name = log['location']
    nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)

    ego_x, ego_y, _ = translation
    yaw = yaw_from_quaternion(nusc.get('ego_pose', sample['data']['LIDAR_TOP'])['rotation'])
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)

    H, W = bev_canvas.shape[:2]
    center = np.array([W // 2, H // 2])

    # Draw lanes and connectors
    for lane in list(nusc_map.drivable_area):
            for poly in lane["polygon_tokens"]:
                polygon = nusc_map.extract_polygon(poly)
                coords = np.array(polygon.exterior.coords)
                dx = coords[:, 0] - ego_x
                dy = coords[:, 1] - ego_y
                rotated_x = cos_y * dx - sin_y * dy
                rotated_y = sin_y * dx + cos_y * dy
                rotated_x /= 1
                rotated_y /= 1
                rotated_y = -rotated_y
                coords[:, 0] = rotated_x * zoom + center[0]
                coords[:, 1] = rotated_y * zoom + center[1]
                coords = coords.astype(np.int32)
                cv2.fillPoly(bev_canvas, [coords], color=1)

    for lane in list(nusc_map.walkway):
        polygon = nusc_map.extract_polygon(lane["polygon_token"])
        coords = np.array(polygon.exterior.coords)
        dx = coords[:, 0] - ego_x
        dy = coords[:, 1] - ego_y
        rotated_x = cos_y * dx - sin_y * dy
        rotated_y = sin_y * dx + cos_y * dy
        rotated_x /= 1
        rotated_y /= 1
        rotated_y = -rotated_y
        coords[:, 0] = rotated_x * zoom + center[0]
        coords[:, 1] = rotated_y * zoom + center[1]
        coords = coords.astype(np.int32)
        cv2.fillPoly(bev_canvas, [coords], color=2)

    for lane in list(nusc_map.lane) + list(nusc_map.lane_connector):
        poses = nusc_map.discretize_lanes([lane["token"]], 4.0)  # returns dict of lists
        for key in poses.keys():
            coords = np.array([[pose[0], pose[1]] for pose in poses[key]])  # stack all points

            # transform to ego frame
            dx = coords[:, 0] - ego_x
            dy = coords[:, 1] - ego_y
            rotated_x = cos_y * dx - sin_y * dy
            rotated_y = sin_y * dx + cos_y * dy
            rotated_y = -rotated_y
            coords[:, 0] = rotated_x * zoom + center[0]
            coords[:, 1] = rotated_y * zoom+ center[1]
            coords = coords.astype(np.int32)

            # draw centerline
            cv2.polylines(bev_canvas, [coords], isClosed=False, color=3, thickness=int(1*zoom))

    annotations = [nusc.get("sample_annotation", ann) for ann in sample["anns"]]
    for ann in annotations:
        x, y, z = ann['translation']
        yaw_obj = yaw_from_quaternion(ann['rotation'])
        length, width, height = ann['size']

        # transform to ego frame
        dx = x - ego_x
        dy = y - ego_y
        rotated_x = cos_y * dx - sin_y * dy
        rotated_y = sin_y * dx + cos_y * dy
        rotated_y = -rotated_y

        # create box corners in object frame
        box = np.array([
            [-width / 2, 0],
            [width / 2, 0],
            [width / 2, length],
            [-width / 2, length]
        ])

        # rotate box by object yaw relative to ego
        c, s = np.cos(yaw_obj - yaw), np.sin(yaw_obj - yaw)
        rot_box = np.zeros_like(box)
        rot_box[:, 0] = c * box[:, 0] - s * box[:, 1]
        rot_box[:, 1] = s * box[:, 0] + c * box[:, 1]

        # apply zoom and center
        rot_box[:, 0] = rot_box[:, 0] * zoom + center[0] + rotated_x * zoom
        rot_box[:, 1] = rot_box[:, 1] * zoom + center[1] + rotated_y * zoom
        rot_box = rot_box.astype(np.int32)

        # choose color based on category
        if ann['category_name'].startswith("vehicle"):
            color = 5
        elif ann['category_name'].startswith("human"):
            color = 6
        elif ann['category_name'] in ["movable_object.trafficcone","movable_object.barrier","movable_object.pushable_pullable"]:
            color = 4
        else:
            print(ann['category_name'])
            continue

        cv2.fillPoly(bev_canvas, [rot_box], color=color)

    # Draw ego box
    ego_length, ego_width = 4, 4
    box = np.array([
        [-ego_width / 2, 0],
        [ego_width / 2, 0],
        [ego_width / 2, ego_length],
        [-ego_width / 2, ego_length]
    ])

    box = box / 1 + center
    box = box.astype(np.int32)
    cv2.fillPoly(bev_canvas, [box], color=5)

    # Rotate the canvas
    bev_canvas = np.rot90(bev_canvas, k=1).copy()

    # Crop to the target BEV size
    target_H, target_W = cfg.bev_semantic_frame
    start_y = (bev_canvas.shape[0] - target_H) // 2
    start_x = (bev_canvas.shape[1] - target_W) // 2
    bev_semantic_map = bev_canvas[start_y:start_y+target_H, start_x:start_x+target_W]
    return torch.tensor(bev_semantic_map)



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