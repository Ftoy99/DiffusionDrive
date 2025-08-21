from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List
import cv2
import numpy as np
import numpy.typing as npt

import torch
from pyquaternion import Quaternion
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor

from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from navsim.agents.hidden.hidden_config import HiddenConfig
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
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
        # output_dir = Path("/mnt/ds/debug")
        # output_dir.mkdir(parents=True, exist_ok=True)
        # tensor_img = features["camera_feature"]  # C,H,W
        # img = tensor_img.permute(1, 2, 0).cpu().numpy()  # H,W,C
        # img = (img * 255).astype('uint8')
        # cv2.imwrite(str(output_dir / "stitched_camera.png"), img[:, :, ::-1])  # RGB→BGR
        features["gaze"] = self._get_gaze_feature(features["camera_feature"])

        features["lidar_feature"] = self._get_lidar_feature(agent_input).cpu().numpy()


        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_driving_command, dtype=torch.float32),
                torch.tensor([agent_input.ego_velocity], dtype=torch.float32),
                torch.tensor([agent_input.ego_acceleration], dtype=torch.float32),
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
            # print("X min/max:", point_cloud[:, 0].min(), point_cloud[:, 0].max())
            # print("Y min/max:", point_cloud[:, 1].min(), point_cloud[:, 1].max())
            # print("Z min/max:", point_cloud[:, 2].min(), point_cloud[:, 2].max())

            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            # print(hist.shape)
            # print(hist)
            # hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            #TODO MUST FIX THIS revert the previous line this was for debug
            overhead_splat = hist / self._config.hist_max_per_pixel
            # overhead_splat = (hist > 0).astype(np.uint8) * 255  # 255 = occupied, 0 = empty
            overhead_splat = overhead_splat.T
            return overhead_splat

        # Remove points above the vehicle
        # print("Before filter:", lidar_pc.shape)
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        # print("After filter:", lidar_pc.shape)
        #
        # print("below Before filter:", lidar_pc.shape)
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        # print("below After filter:", lidar_pc.shape)
        #
        # print("above Before filter:", lidar_pc.shape)
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        # print("above After filter:", above.shape)
        #
        # print("above_features max/min:", above.max(), above.min())
        above_features = splat_points(above)
        # print("above_features max/min:", above_features.max(), above_features.min())
        # print("above After splatting:", above.shape)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)

        # full_bins = np.count_nonzero(features)
        # print("Number of full bins:", full_bins)

        bev_img = features[:, :, 0]  # single channel
        cv2.imwrite("/mnt/ds/debug/lidar_bev_img.png", bev_img)
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
        return "transfuser_target"

    def compute_targets(self, data: NuTargetData) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = data.trajectory

        ego_pose = StateSE2(data.ego_pose_global_cords[0],data.ego_pose_global_cords[0],data.ego_pose_heading)

        agent_states, agent_labels = self._compute_agent_targets(data.annotations,data)
        bev_semantic_map = self._compute_bev_semantic_map(data,data.map,data.map_api,ego_pose)

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

    def _compute_bev_semantic_map(
            self, annotations, map : NuScenesMap, map_api: NuScenesMapExplorer, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)

        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map,map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map,map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
            self,map:NuScenesMap, map_api: NuScenesMapExplorer, ego_pose: StateSE2, layers: List[SemanticMapLayer]
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
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
            self,map:NuScenesMap, map_api: NuScenesMapExplorer, ego_pose: StateSE2, layers: List[SemanticMapLayer]
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
        patch_box = (
        ego_pose.point.x, ego_pose.point.y, self._config.bev_radius, self._config.bev_radius)  # (xc, yc, w, h)
        # Query objects that intersect with patch
        map_objects = map_api.get_records_in_patch(
            patch_box, mode='intersect'
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in map_objects:
            for map_object in map_objects[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(self, data, layers: TrackedObjectType) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for ann in data.annotations:
            category = ann['category_name']
            if category not in NameMapping:
                continue
            agent_type = tracked_object_types[NameMapping[ann["category_name"]]]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = ann["translation"][0], ann["translation"][1] ,Quaternion(ann["rotation"]).yaw_pitch_roll[0]
                box_length, box_width, box_height = ann["size"][0],ann["size"][1],ann["size"][2]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

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
