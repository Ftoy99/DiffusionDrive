import gzip
import os

import cv2
import numpy as np
import torch
import pickle

def draw_bev(features, ego_fut_trajs_rel, trajectories=None,boxes=None, path="/mnt/ds/debug/original/bev.png"):
    bev_img_draw = features[0, :, :]  # (H,W)
    bev_img_draw = (bev_img_draw * 255).to(torch.uint8).cpu().numpy()

    # convert grayscale to 3-channel RGB
    bev_img_draw = cv2.cvtColor(bev_img_draw, cv2.COLOR_GRAY2BGR)

    H, W, _ = bev_img_draw.shape
    resolution = 0.1
    origin = np.array([W // 2, H // 2])

    # Draw ego/vehicle future trajectory (dark blue)
    if ego_fut_trajs_rel is not None:
        for pos in ego_fut_trajs_rel:
            x_px = int(origin[0] - pos[1] / resolution)
            y_px = int(origin[1] - pos[0] / resolution)
            if 0 <= x_px < W and 0 <= y_px < H:
                bev_img_draw[y_px, x_px] = (0, 0, 139)  # Dark blue BGR

        # Draw other trajectories (green) and boxes (yellow filled)
        if trajectories is not None and boxes is not None:
            for traj_list, box_list in zip(trajectories, boxes):
                for pos, box_corners_px in zip(traj_list, box_list):
                    # Draw trajectory point
                    x_px = int(origin[0] - pos[1] / resolution)
                    y_px = int(origin[1] - pos[0] / resolution)
                    if 0 <= x_px < W and 0 <= y_px < H:
                        bev_img_draw[y_px, x_px] = (0, 255, 0)  # Green

                        # Draw filled rotated box
                        pts = box_corners_px.reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(bev_img_draw, [pts], color=(0, 255, 255))  # Yellow fill

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bev_img_draw)


def draw_semantic(bev_map_tensor, ego_fut_trajs_rel=None, trajectories=None,boxes=None, save_path="/mnt/ds/debug/original/bev_semantic_debug.png"):
    bev_map = bev_map_tensor.cpu().numpy().astype(np.int32)
    H, W = bev_map.shape

    colors = {
        0: (0, 0, 0),        # background → Black
        1: (128, 64, 128),   # road → Dark Purple
        2: (0, 255, 0),      # walkway → Green
        3: (255, 255, 0),    # centerline → Yellow
        4: (192, 192, 192),  # static object → Light Gray
        5: (0, 0, 255),      # vehicle → Blue
        6: (255, 0, 0),      # pedestrian → Red
    }

    bev_img = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in colors.items():
        bev_img[bev_map == label] = color

    # angle_deg = 90  # rotation angle in degrees (positive = CCW)
    # center = (bev_img.shape[1] // 2, bev_img.shape[0] // 2)  # rotate around image center
    # M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # bev_img = cv2.warpAffine(bev_img, M, (bev_img.shape[1], bev_img.shape[0]))
    # # Flip vertically (upside down)
    # bev_img = cv2.flip(bev_img, 0)  # 0 = vertical flip

    origin = np.array([W // 2, H // 2])
    # Draw ego/vehicle trajectory (dark blue)
    if ego_fut_trajs_rel is not None:
        for pos in ego_fut_trajs_rel:
            x_px = int(origin[0] - pos[1])
            y_px = int(origin[1] - pos[0])
            if 0 <= x_px < W and 0 <= y_px < H:
                bev_img[y_px, x_px] = (0, 0, 139)  # Dark blue

    # Draw other trajectories (green)
    if trajectories is not None:
        for traj_list, box_list in zip(trajectories, boxes):
            for pos, box in zip(traj_list, box_list):
                x_px = int(origin[0] - pos[1])
                y_px = int(origin[1] - pos[0])
                if 0 <= x_px < W and 0 <= y_px < H:
                    bev_img[y_px, x_px] = (0, 255, 0)  # Green

                    # polygon points
                    box_arr = np.array(box)  # shape (5, 1, 2)
                    box_pts = box_arr.reshape(-1, 2)  # shape (5, 2)
                    cv2.fillPoly(bev_img, [box_pts.astype(np.int32)], color=(0, 255, 255))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bev_img)

def draw_feature_image(feature_tensor, save_path="/mnt/ds/debug/original/feature_debug.png"):
    """
    Save a 3xHxW image tensor as a PNG for debugging.
    :param feature_tensor: torch.Tensor (3, H, W)
    """
    img = feature_tensor.cpu().numpy()
    # Convert C,H,W -> H,W,C
    img = np.transpose(img, (1, 2, 0))
    # If float in [0,1], scale to 0-255
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

if __name__ == "__main__":
    feat_path = "/mnt/ds/navsim-main/exp/training_cache/2021.05.12.22.00.38_veh-35_01008_01518/0ce0aa336fe751a4/transfuser_feature.gz"
    target_path = "/mnt/ds/navsim-main/exp/training_cache/2021.05.12.22.00.38_veh-35_01008_01518/0ce0aa336fe751a4/transfuser_target.gz"

    with gzip.open(feat_path, "rb") as f:
        features = pickle.load(f)

    with gzip.open(target_path, "rb") as f:
        target = pickle.load(f)

    draw_bev(features["lidar_feature"],target["trajectory"],features['trajectories'],features['boxes'])
    draw_semantic(target["bev_semantic_map"],target["trajectory"],features['trajectories'],features['boxes'])
    draw_feature_image(features["camera_feature"])