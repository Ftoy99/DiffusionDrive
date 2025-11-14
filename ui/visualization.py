import base64

import cv2
import numpy as np

def draw_semantic(bev_map_tensor):
    bev_map = bev_map_tensor.cpu().numpy().astype(np.int32)
    H, W = bev_map.shape
    colors = {
        0: (0, 0, 0),  # background → Black
        1: (128, 64, 128),  # road → Dark Purple
        2: (255, 192, 203), # pedestrian crossing
        3: (0, 255, 0),  # walkway → Green
        4: (255, 255, 0),  # centerline → Yellow
        5: (192, 192, 192),  # static object → Light Gray
        6: (0, 0, 255),  # vehicle → Blue
        7: (255, 0, 0),  # pedestrian → Red
    }
    bev_img = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in colors.items():
        bev_img[bev_map == label] = color
    return bev_img


def img_to_base64(img):
    # img: numpy array (H, W, C), BGR
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')