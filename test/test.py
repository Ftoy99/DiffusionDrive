import os

import cv2
import numpy as np
import torch

def draw_bev(features,ego_fut_trajs_rel,path="/mnt/ds/debug/original/bev.png"):
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


if __name__ == '__main__':
    import gzip
    import pickle

    path = "/mnt/jimmys/DiffusionDrive/navsim/exp/training_cache/2021.08.31.17.42.52_veh-40_01331_01444/0d580b50789c5fb8/transfuser_target.gz"

    with gzip.open(path, "rb") as f:
        data = pickle.load(f)

    print(type(data))
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
    elif isinstance(data, (list, tuple)):
        print("Length:", len(data))
        print("First element type:", type(data[0]))

    future_traj = data["trajectory"]
    print(data["bev_semantic_map"].shape)
    draw_semantic(data["bev_semantic_map"])

    path = "/mnt/jimmys/DiffusionDrive/navsim/exp/training_cache/2021.08.31.17.42.52_veh-40_01331_01444/0d580b50789c5fb8/transfuser_feature.gz"
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    draw_bev(data["lidar_feature"],future_traj)