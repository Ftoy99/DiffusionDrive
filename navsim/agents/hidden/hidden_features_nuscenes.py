from enum import IntEnum
from typing import Any, Dict, List
import cv2
import numpy as np
import numpy.typing as npt
import pyquaternion

import torch
from pyquaternion import Quaternion
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor

from shapely import affinity

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.state_representation import StateSE2

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from navsim.agents.hidden.hidden_config import HiddenConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from navsim.agents.hidden.depth_gaze import depth_inf

front_cameras = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"]
NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "vehicle",
    "vehicle.bus.bendy": "vehicle",
    "vehicle.bus.rigid": "vehicle",
    "vehicle.car": "vehicle",
    "vehicle.construction": "vehicle",
    "vehicle.motorcycle": "vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "vehicle",
    "vehicle.truck": "vehicle",
}


class NuFeatureData:

    def __init__(self):
        self.images = {}
        self.lidar = None
        self.ego_driving_command = None
        self.ego_velocity = None
        self.ego_acceleration = None
        self.token = None


class NuTargetData:

    def __init__(self):
        self.trajectory = None
        self.annotations = []
        self.ego_pose_global_cords = None
        self.ego_pose_heading = None
        self.map_api = None
        self.map = None


class HiddenFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self, config: HiddenConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"

    def compute_features(self, agent_input: NuFeatureData) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}
        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["gaze"] = self._get_gaze_feature(features["camera_feature"])
        features["lidar_feature"] = self._get_lidar_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

    def _get_camera_feature(self, agent_input: NuFeatureData) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        # Crop to ensure 4:1 aspect ratio
        l0 = agent_input.images["CAM_FRONT_LEFT"][28:-28, 416:-416]
        f0 = agent_input.images["CAM_FRONT"][28:-28]
        r0 = agent_input.images["CAM_FRONT_RIGHT"][28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        # resized_image = cv2.resize(stitched_image, (2048, 512))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

    def _get_lidar_feature(self, agent_input: NuFeatureData) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidar

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)

        # --- rotate & flip before transpose ---
        features = np.rot90(features, k=-1, axes=(0, 1))  # 90° clockwise
        features = np.flip(features, axis=1)  # horizontal flip
        features = np.ascontiguousarray(features)  # ensure contiguous for OpenCV

        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)

    def _get_gaze_feature(self, image):
        C, H, W = image.shape

        # Crop the image to remove asphalt
        crop_H = int(H * 0.75)
        img_cropped = image[:, :crop_H, :]

        # Depth inference
        depth = img_cropped
        depth = depth_inf(ToPILImage()(depth))
        depth_tensor = ToTensor()(depth)

        # Estimate gaze from depth
        gaze_x, gaze_y = self._estimate_gaze_from_depth(depth_tensor)

        # Map gaze_y back to original image height
        gaze_y = gaze_y * (crop_H / depth_tensor.shape[1])  # corrected shape index

        # Crop around gaze point
        crop_size = 144
        x1 = int(gaze_x - crop_size // 2)
        y1 = int(gaze_y - crop_size // 2)

        x1 = max(0, min(x1, W - crop_size))
        y1 = max(0, min(y1, H - crop_size))  # H is still original height for full image crop
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        gaze_crop = image[:, y1:y2, x1:x2]
        return gaze_crop  # optionally: return gaze_x, gaze_y

    def _estimate_gaze_from_depth(self, depthImg, top_percent=0.05):
        _, H, W = depthImg.shape
        k = int(H * W * top_percent)

        # Flatten and get top-k closest points (smallest depth)
        depth_flat = depthImg.view(-1)
        topk_vals, topk_idx = torch.topk(-depth_flat, k)

        ys = topk_idx // W
        xs = topk_idx % W

        gaze_x = xs.float().mean()
        gaze_y = ys.float().mean()
        return gaze_x.item(), gaze_y.item()


class HiddenTargetBuilder(AbstractTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(self, config: HiddenConfig):
        """
        Initializes target builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_tafrget"

    def compute_targets(self, data: NuTargetData, nusc, sample) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = np.array([[tr.x, tr.y, tr.heading] for tr in data.trajectory], dtype=np.float32)
        trajectory = torch.tensor(trajectory, dtype=torch.float32)

        ego_pose = StateSE2(data.ego_pose_global_cords[0],data.ego_pose_global_cords[1],data.ego_pose_heading)
        agent_states, agent_labels = self._compute_agent_targets(data.annotations,data)
        bev_semantic_map = self._compute_bev_semantic_map(data,data.map,data.map_api,ego_pose,nusc,sample)

        return {
            "trajectory": trajectory, # x y heading
            "agent_states": agent_states, # Agents in 5 dim tensor
            "agent_labels": agent_labels, # Tree false mask
            "bev_semantic_map": bev_semantic_map,
        }

    def _compute_agent_targets(self, annotations,data):
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """
        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: HiddenConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (config.lidar_min_y <= y <= config.lidar_max_y)

        for annotation in annotations:
            box_x, box_y, box_heading, box_length, box_width = (
                annotation["translation"][0] - data.ego_pose_global_cords[0],
                annotation["translation"][1] - data.ego_pose_global_cords[1],
                Quaternion(annotation["rotation"]).yaw_pitch_roll[0],
                annotation["size"][0],
                annotation["size"][1],
            )
            category = annotation['category_name']
            if category not in NameMapping:
                continue
            if NameMapping[category] == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32))

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def get_ego_pose(self,sample, nusc):
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])

        # global location
        translation = ego_pose['translation']  # [x, y, z] in meters
        rotation = ego_pose['rotation']  # quaternion [w, x, y, z]

        return translation, rotation

    def yaw_from_quaternion(self,q):
        quat = pyquaternion.Quaternion(q)
        return quat.yaw_pitch_roll[0]

    def _compute_bev_semantic_map(
            self, data, map : NuScenesMap, map_api: NuScenesMapExplorer, ego_pose: StateSE2 , nusc , sample
    ) -> torch.Tensor:
        cfg = HiddenConfig()
        zoom = 4.0  # 2× zoom, increase to zoom in more
        # Make a large square canvas to avoid clipping after rotation
        max_dim = max(cfg.bev_semantic_frame)
        bev_canvas = np.zeros((max_dim, max_dim), dtype=np.uint8)
        translation, rotation = self.get_ego_pose(sample, nusc)

        # Get map name from sample
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        map_name = log['location']
        nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)

        ego_x, ego_y, _ = translation
        yaw = self.yaw_from_quaternion(nusc.get('ego_pose', sample['data']['LIDAR_TOP'])['rotation'])
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
            poses = nusc_map.discretize_lanes([lane["token"]], 3.0)  # returns dict of lists
            for key in poses.keys():
                coords = np.array([[pose[0], pose[1]] for pose in poses[key]])  # stack all points

                # transform to ego frame
                dx = coords[:, 0] - ego_x
                dy = coords[:, 1] - ego_y
                rotated_x = cos_y * dx - sin_y * dy
                rotated_y = sin_y * dx + cos_y * dy
                rotated_y = -rotated_y
                coords[:, 0] = rotated_x * zoom + center[0]
                coords[:, 1] = rotated_y * zoom + center[1]
                coords = coords.astype(np.int32)

                # draw centerline
                cv2.polylines(bev_canvas, [coords], isClosed=False, color=3, thickness=int(1 * zoom))

        annotations = [nusc.get("sample_annotation", ann) for ann in sample["anns"]]
        for ann in annotations:
            x, y, z = ann['translation']
            yaw_obj = self.yaw_from_quaternion(ann['rotation'])
            length, width, height = ann['size']

            # box corners in object local frame (centered at origin)
            box = np.array([
                [-width / 2, -length / 2],
                [width / 2, -length / 2],
                [width / 2, length / 2],
                [-width / 2, length / 2]
            ])

            # rotate to global frame with object yaw
            c, s = np.cos(yaw_obj), np.sin(yaw_obj)
            rot_box = np.zeros_like(box)
            rot_box[:, 0] = c * box[:, 0] - s * box[:, 1]
            rot_box[:, 1] = s * box[:, 0] + c * box[:, 1]

            # translate to object center (global coords)
            rot_box[:, 0] += x
            rot_box[:, 1] += y

            # --- now same as walkway ---
            dx = rot_box[:, 0] - ego_x
            dy = rot_box[:, 1] - ego_y
            rotated_x = cos_y * dx - sin_y * dy
            rotated_y = sin_y * dx + cos_y * dy
            rotated_y = -rotated_y
            rot_box[:, 0] = rotated_x * zoom + center[0]
            rot_box[:, 1] = rotated_y * zoom + center[1]
            rot_box = rot_box.astype(np.int32)

            # color logic same as before
            if ann['category_name'].startswith("vehicle"):
                color = 5
            elif ann['category_name'].startswith("human"):
                color = 6
            elif ann['category_name'] in ["movable_object.trafficcone", "movable_object.barrier",
                                          "movable_object.pushable_pullable"]:
                color = 4
            else:
                continue

            cv2.fillPoly(bev_canvas, [rot_box], color=color)

        # # Draw ego box
        # ego_length, ego_width = 4, 4
        # box = np.array([
        #     [-ego_width / 2, 0],
        #     [ego_width / 2, 0],
        #     [ego_width / 2, ego_length],
        #     [-ego_width / 2, ego_length]
        # ])
        #
        # box = box / 1 + center
        # box = box.astype(np.int32)
        # cv2.fillPoly(bev_canvas, [box], color=5)

        # Rotate the canvas
        bev_canvas = np.rot90(bev_canvas, k=1).copy()

        # Crop to the target BEV size
        target_H, target_W = cfg.bev_semantic_frame
        start_y = (bev_canvas.shape[0] - target_H) // 2
        start_x = (bev_canvas.shape[1] - target_W) // 2
        bev_semantic_map = bev_canvas[start_y:start_y + target_H, start_x:start_x + target_W]
        return torch.tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
            self,map:NuScenesMap, map_api: NuScenesMapExplorer,bounds, ego_pose: StateSE2
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: binary mask as numpy array
        """
        # Create the mask
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)

        x_min, y_min, x_max, y_max = bounds
        pts = np.array([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min]
        ], dtype=np.float32)

        # Translate to ego frame
        pts -= np.array([ego_pose.point.x, ego_pose.point.y])
        c, s = np.cos(-ego_pose.heading), np.sin(-ego_pose.heading)
        R = np.array([[c, -s], [s, c]])
        pts_local = pts @ R.T

        # Convert to pixel coordinates
        scale = self._config.bev_pixel_height / (2 * self._config.bev_radius)
        pts_pix = ((pts_local + self._config.bev_radius) * scale).astype(np.int32)

        # Clip to frame
        pts_pix[:, 0] = np.clip(pts_pix[:, 0], 0, self._config.bev_pixel_width - 1)
        pts_pix[:, 1] = np.clip(pts_pix[:, 1], 0, self._config.bev_pixel_height - 1)


        cv2.fillPoly(map_polygon_mask, [pts_pix], color=255)

        # Rotate/flip to match BEV convention
        mask = np.rot90(map_polygon_mask)[::-1]
        return mask > 0

    def _compute_map_linestring_mask(
            self,map:NuScenesMap, map_api: NuScenesMapExplorer,bounds, ego_pose: StateSE2
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        # map_object_dict = map_api.get_proximal_map_objects(
        #     point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        # )
        # Create the mask
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)

        x_min, y_min, x_max, y_max = bounds
        pts = np.array([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min]
        ], dtype=np.float32)

        # Translate to ego frame
        pts -= np.array([ego_pose.point.x, ego_pose.point.y])
        c, s = np.cos(-ego_pose.heading), np.sin(-ego_pose.heading)
        R = np.array([[c, -s], [s, c]])
        pts_local = pts @ R.T

        # Convert to pixel coordinates
        scale = self._config.bev_pixel_height / (2 * self._config.bev_radius)
        pts_pix = ((pts_local + self._config.bev_radius) * scale).astype(np.int32)

        # Clip to frame
        pts_pix[:, 0] = np.clip(pts_pix[:, 0], 0, self._config.bev_pixel_width - 1)
        pts_pix[:, 1] = np.clip(pts_pix[:, 1], 0, self._config.bev_pixel_height - 1)

        # For linestring
        pts = np.array(pts_pix, dtype=np.int32)
        cv2.polylines(map_polygon_mask, [pts], isClosed=False, color=255, thickness=1)

        # Rotate/flip to match BEV convention
        mask = np.rot90(map_polygon_mask)[::-1]
        return mask > 0

    def _compute_box_mask(self, ann, ego) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)

        # Extract box info
        x, y = ann["translation"][0], ann["translation"][1]
        length, width = ann["size"][0], ann["size"][1]
        heading = Quaternion(ann["rotation"]).yaw_pitch_roll[0]

        # Compute rectangle corners in global frame
        dx = length / 2
        dy = width / 2
        corners = np.array([
            [dx, dy],
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy]
        ])
        # Rotate by heading
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]])
        corners_rot = corners @ R.T
        # Translate to global position
        corners_rot += np.array([x, y])

        # Transform to ego frame
        corners_rot -= np.array([ego.point.x, ego.point.y])
        c, s = np.cos(-ego.heading), np.sin(-ego.heading)
        R = np.array([[c, -s], [s, c]])
        corners_local = corners_rot @ R.T

        # Convert to pixel coordinates
        scale = self._config.bev_pixel_height / (2 * self._config.bev_radius)
        pts_pix = ((corners_local + self._config.bev_radius) * scale).astype(np.int32)
        pts_pix[:, 0] = np.clip(pts_pix[:, 0], 0, self._config.bev_pixel_width - 1)
        pts_pix[:, 1] = np.clip(pts_pix[:, 1], 0, self._config.bev_pixel_height - 1)

        # Draw filled rectangle
        cv2.fillPoly(map_polygon_mask, [pts_pix], color=255)

        # Rotate/flip to match BEV convention
        mask = np.rot90(map_polygon_mask)[::-1]

        return mask > 0

    @staticmethod
    def _query_map_objects(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self, layers=layers)
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in TransFuser."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
