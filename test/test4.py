import gzip
import os

import cv2
import numpy as np
import torch
import lzma
import pickle

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

def draw_semantic(bev_map_tensor,trajectories = None, save_path="/mnt/ds/debug/original/bev_semantic_debug.png"):
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

    # Draw trajectories
    # Paint trajectory points pink
    if trajectories is not None:
        pixel_center = np.array([[0, 256 / 2.0]])
        for traj in trajectories:
            traj_pixels = (np.array(traj).reshape(-1, 2)) + pixel_center
            traj_pixels = traj_pixels.astype(np.int32)

            for x, y in traj_pixels:
                if 0 <= x < W and 0 <= y < H:  # safety check
                    bev_img[x, y] = (255, 105, 180)  # Pink pixel



    # Save image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bev_img)


if __name__ == "__main__":
    feat_path = "/mnt/ds/navsim-main/exp/training_cache/2021.05.12.23.36.44_veh-35_00152_00504/2d9168675ce355a2/transfuser_feature.gz"
    target_path = "/mnt/ds/navsim-main/exp/training_cache/2021.05.12.23.36.44_veh-35_00152_00504/2d9168675ce355a2/transfuser_target.gz"

    with gzip.open(feat_path, "rb") as f:
        features = pickle.load(f)

    with gzip.open(target_path, "rb") as f:
        target = pickle.load(f)


    print(type(features))
    print(type(target))

    draw_semantic(target["bev_semantic_map"],features['trajectories'])
