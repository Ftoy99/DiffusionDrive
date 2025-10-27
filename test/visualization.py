import gzip
import os

import cv2
import numpy as np
import torch
import pickle

from navsim.agents.hidden_v2.hidden_config import HiddenConfig

def draw_bev(features, ego_fut_trajs_rel, trajectories=None, boxes=None):
    bev_img_draw = features[0, :, :]  # (H,W)
    bev_img_draw = (bev_img_draw * 255).to(torch.uint8).cpu().numpy()

    # convert grayscale to 3-channel RGB
    bev_img_draw = cv2.cvtColor(bev_img_draw, cv2.COLOR_GRAY2BGR)

    H, W, _ = bev_img_draw.shape
    resolution = 0.1
    origin = np.array([W // 2, H // 2])

    # Draw ego/vehicle future trajectory (green, thicker)
    if ego_fut_trajs_rel is not None:
        for pos in ego_fut_trajs_rel:
            x_px = int(origin[0] - pos[1] / resolution)
            y_px = int(origin[1] - pos[0] / resolution)
            # flip both axes
            x_px = W - 1 - x_px
            y_px = H - 1 - y_px
            if 0 <= x_px < W and 0 <= y_px < H:
                # Draw a small square instead of single pixel
                size = 2  # box half-size
                x_min = max(x_px - size, 0)
                x_max = min(x_px + size + 1, W)
                y_min = max(y_px - size, 0)
                y_max = min(y_px + size + 1, H)
                bev_img_draw[y_min:y_max, x_min:x_max] = (0, 255, 0)  # green BGR

    # Draw a unique color block in the middle
    center_size = 2  # half-size, so total 4x4
    x_c, y_c = origin
    x_min = max(x_c - center_size, 0)
    x_max = min(x_c + center_size + 1, W)
    y_min = max(y_c - center_size, 0)
    y_max = min(y_c + center_size + 1, H)
    bev_img_draw[y_min:y_max, x_min:x_max] = (255, 0, 255)  # magenta

    return bev_img_draw


def draw_bev_save(features, ego_fut_trajs_rel, trajectories=None,boxes=None, path="/mnt/ds/debug/original/bev.png"):
    bev_img_draw = draw_bev(features,ego_fut_trajs_rel,trajectories,boxes)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bev_img_draw)


def draw_semantic_save(bev_map_tensor, ego_fut_trajs_rel=None, trajectories=None,boxes=None, save_path="/mnt/ds/debug/original/bev_semantic_debug.png"):
    bev_img = draw_semantic(bev_map_tensor, ego_fut_trajs_rel, trajectories)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bev_img)


def draw_semantic(bev_map_tensor, ego_fut_trajs_rel, trajectories):
    bev_map = bev_map_tensor.cpu().numpy().astype(np.int32)
    H, W = bev_map.shape
    center = np.array([W / 2, H / 2])
    resolution = 0.1
    colors = {
        0: (0, 0, 0),  # background → Black
        1: (128, 64, 128),  # road → Dark Purple
        2: (0, 255, 0),  # walkway → Green
        3: (255, 255, 0),  # centerline → Yellow
        4: (192, 192, 192),  # static object → Light Gray
        5: (0, 0, 255),  # vehicle → Blue
        6: (255, 0, 0),  # pedestrian → Red
    }
    bev_img = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in colors.items():
        bev_img[bev_map == label] = color

    if ego_fut_trajs_rel is not None:
        for pos in ego_fut_trajs_rel:

            x_px = int(center[0] - pos[1] / resolution)
            y_px = int(center[1] - pos[0] / resolution)
            if 0 <= x_px < W and 0 <= y_px < H:
                # Draw a small square instead of single pixel
                size = 2  # box half-size
                x_min = max(x_px - size, 0)
                x_max = min(x_px + size + 1, W)
                y_min = max(y_px - size, 0)
                y_max = min(y_px + size + 1, H)
                bev_img[y_min:y_max, x_min:x_max] = (0, 255, 0)  # green BGR

    # Draw other trajectories (colored dots)
    if trajectories is not None:
        for trajectory in trajectories:
            color = (0,255,255)
            for waypoint in trajectory:
                pos_np = waypoint[:2].cpu().numpy() # shape (1,2)

                # pos_np = flip_coords(pos_np, H)
                # pos_np = flip_upside_down(pos_np, H)

                x_px = int(center[0] - pos_np[0]/ resolution)
                y_px = int(center[1] - pos_np[1]/ resolution)
                # flip both axes
                x_px = W - 1 - x_px
                y_px = H - 1 - y_px
                if 0 <= x_px < W and 0 <= y_px < H:
                    size = 1
                    x_min = max(x_px - size, 0)
                    x_max = min(x_px + size + 1, W)
                    y_min = max(y_px - size, 0)
                    y_max = min(y_px + size + 1, H)
                    bev_img[y_min:y_max, x_min:x_max] = color

    # Draw a unique color block in the middle
    center_size = 2  # half-size, so total 4x4
    x_c, y_c = center
    x_c = int(x_c)
    y_c = int(y_c)
    x_min = max(x_c - center_size, 0)
    x_max = min(x_c + center_size + 1, W)
    y_min = max(y_c - center_size, 0)
    y_max = min(y_c + center_size + 1, H)
    bev_img[y_min:y_max, x_min:x_max] = (255, 0, 255)  # magenta
    return bev_img


def flip_upside_down(coords, H):
    """
    coords: (N,2) array (x, y)
    H: image height
    """
    x, y = coords[:,0], coords[:,1]
    new_x = x
    new_y = H - 1 - y
    return np.stack([new_x, new_y], axis=-1)


def flip_coords(coords,W):
    x, y = coords[:,0], coords[:,1]
    new_x = y
    new_y = W - 1 - x
    return np.stack([new_x, new_y], axis=-1)

def rotate_coords(coords, angle, center):
    """
    coords: (N,2) array
    angle: radians, CCW
    center: (cx, cy)
    """
    shifted = coords - center
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    rotated = shifted @ R.T
    return rotated + center

def box_to_corners_px_manual(box, bev_pixel_width, bev_pixel_size):
    x, y, heading, length, width, _, _ = box
    c, s = np.cos(heading), np.sin(heading)
    dx = length / 2
    dy = width / 2
    # corners in local frame
    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy]
    ])
    # rotate & translate
    corners = (corners @ np.array([[c, -s], [s, c]]).T) + np.array([x, y])
    # convert to pixel indices manually
    pixel_center = np.array([0, bev_pixel_width / 2.0])
    corners_px = (corners / bev_pixel_size) + pixel_center
    return corners_px.astype(np.int32)


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
    feat_path = "/mnt/ds/navsim-main/exp/training_cache/2021.05.12.22.00.38_veh-35_01008_01518/41bb74b4738f5a8b/transfuser_feature.gz"
    target_path = "/mnt/ds/navsim-main/exp/training_cache/2021.05.12.22.00.38_veh-35_01008_01518/41bb74b4738f5a8b/transfuser_target.gz"

    with gzip.open(feat_path, "rb") as f:
        features = pickle.load(f)

    with gzip.open(target_path, "rb") as f:
        target = pickle.load(f)

    draw_bev_save(features["lidar_feature"],target["trajectory"],features['trajectories'],None)
    draw_semantic(target["bev_semantic_map"],target["trajectory"],features['trajectories'],None)
    draw_feature_image(features["camera_feature"])