import base64
import gzip
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import hydra
import torch
from PIL import Image
from hydra.utils import instantiate
from omegaconf import DictConfig
from flask import Flask, render_template, request, send_file, Response, jsonify

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.hidden_v2.hidden_config import HiddenConfig
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
import os

from ui.visualization import img_to_base64, draw_semantic

CHECKPOINT_ROOT = "/mnt/ds/navsim-main/exp"
logger = logging.getLogger(__name__)

app = Flask(__name__)
CONFIG_PATH = "../navsim/planning/script/config/training"
CONFIG_NAME = "default_training"

# Global var to hold loader
scene_loader: Optional[SceneLoader] = None
agent: Optional[AbstractAgent] = None
options = []
outputs = None

features = None
targets = None
agent_input = None

# --- Global condition flags ---
use_neighbors = True
use_gaze = True

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scenarios")
def scenarios():
    if scene_loader is None:
        return '<option value="">No scenarios available</option>'

    return "".join(
        f'<option value="{token}">{token}</option>'
        for token in scene_loader.tokens[:10000]
    )

@app.route("/models")
def models():
    global options
    if not options:
        for root, dirs, files in os.walk(CHECKPOINT_ROOT):
            for f in files:
                if f.endswith(".ckpt"):  # checkpoint files
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, CHECKPOINT_ROOT)  # relative path
                    options.append(f'<option value="{full_path}">{rel_path}</option>')

    if not options:
        return '<option value="">No models found</option>'

    return "\n".join(options)

@app.route("/scenario")
def scenario():
    global scene_loader
    global features
    global targets
    global agent_input
    if scene_loader is None:
        return "<div>No SceneLoader initialized</div>"

    # Get the token from request args
    token = request.args.get("scenario")
    if not token:
        return "<div>No scenario selected</div>"

    scene = scene_loader.get_scene_from_token(token)
    agent_input = scene_loader.get_agent_input_from_token(token)

    features = agent.get_feature_builders()[0].compute_features(agent_input)
    targets = agent.get_target_builders()[0].compute_targets(scene)
    return Response(status=200)

@app.route("/camera")
def camera():
    img_tag = f'<img src="/camera_png?_={time.time()}" class="w-full h-32 object-contain">'
    return img_tag

@app.route("/camera_png")
def camera_png():
    global features
    if features is None:
        return "<div>No features initialized</div>"

    img_tensor = features["camera_feature"]
    img_array = (img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

@app.route("/gaze")
def gaze():
    img_tag = f'<img src="/gaze_png?_={time.time()}" class="w-full h-32 object-contain">'
    return img_tag

@app.route("/gaze_png")
def gaze_png():
    global features
    if features is None:
        return "<div>No features initialized</div>"

    img_tensor = features["gaze"]
    img_array = (img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

# @app.route("/lidar")
# def lidar():
#     img_tag = f'<img src="/lidar_png?_={time.time()}" class="w-full h-32 object-contain">'
#     return img_tag
#
# @app.route("/lidar_png")
# def lidar_png():
#     global features
#     if features is None:
#         return "<div>No features initialized</div>"
#
#     img_tensor = features["lidar_feature"]
#     img_array = draw_bev(img_tensor,targets["trajectory"])
#
#     pil_img = Image.fromarray(img_array)
#     buffer = io.BytesIO()
#     pil_img.save(buffer, format='PNG')
#     buffer.seek(0)
#     return send_file(buffer, mimetype='image/png')

@app.route("/semantic-result")
def semantic_result():
    img_tag = f'<img src="/semantic-result-png?_={time.time()}" class="w-full h-32 object-contain">'
    return img_tag

@app.route("/semantic-result-png")
def semantic_result_png():
    global outputs
    if outputs is None:
        return "<div>No features initialized</div>"

    img_tensor = outputs["bev_semantic_map"].squeeze(0).argmax(dim=0)
    ego_trajectory = outputs["trajectory"].squeeze(0)
    img_array = draw_semantic(img_tensor,ego_trajectory,features["trajectories"])

    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

@app.route("/semantic")
def semantic():
    img_tag = f'<img src="/semantic_png?_={time.time()}" class="w-full h-32 object-contain">'
    return img_tag

@app.route("/semantic_png")
def semantic_png():
    global features
    if features is None:
        return "<div>No features initialized</div>"

    img_tensor = targets["bev_semantic_map"]
    img_array = draw_semantic(img_tensor,targets["trajectory"],features["trajectories"])

    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

@app.route("/model_select", methods=["POST"])
def model_select():
    global agent
    model_path = request.form.get("model")
    if not model_path:
        return Response("No model provided", status=400)

    if agent.checkpoint_path != model_path:
        logger.info(f"Switching model to {model_path}")
        agent.checkpoint_path = model_path
        agent.init_from_pretrained()
        logger.info(f"Switching model to {model_path} - END")
        agent.eval()
    return Response(status=200)

@app.route("/use_neighbors", methods=["POST"])
def use_neighbors_endpoint():
    global use_neighbors
    use_neighbors = request.form.get("use_neighbors", "false") in ["true", "True", "1", "on"]
    logger.info(f"Set use_neighbors = {use_neighbors}")
    return Response(status=200)

@app.route("/use_gaze", methods=["POST"])
def use_gaze_endpoint():
    global use_gaze
    use_gaze = request.form.get("use_gaze", "false") in ["true", "True", "1", "on"]
    logger.info(f"Set use_gaze = {use_gaze}")
    return Response(status=200)

@app.route("/scenario_data", methods=["GET"])
def scenario_data():
    logger.info("Get scenario_data")
    global features, targets, agent_input, agent, outputs , use_gaze , use_neighbors
    config = HiddenConfig()

    # --- Camera image ---
    img_tensor = features["camera_feature"]
    img_array = (img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    img_tensor = features["gaze"]
    img_array = (img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    gaze_img_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # --- LiDAR ---
    lidar_pc = agent_input.lidars[-1].lidar_pc
    lidar_pc = lidar_pc[:, lidar_pc[2, :] > config.lidar_split_height]
    mask = (
        (lidar_pc[0, :] >= config.lidar_min_x) & (lidar_pc[0, :] <= config.lidar_max_x) &
        (lidar_pc[1, :] >= config.lidar_min_y) & (lidar_pc[1, :] <= 16)
    )
    lidar_pc = lidar_pc[:, mask]
    lidar_list = lidar_pc.T.tolist()

    # --- Ground truth trajectories ---
    trajectories_tensor = features["trajectories"].cpu().numpy().tolist()
    true_trajectory = targets["trajectory"].cpu().tolist()

    # --- Target vehicle agent boxes ---
    vehicle_bboxes = targets["vehicle_agent_states"].cpu().numpy().tolist()
    vehicle_bboxes_lb = targets["vehicle_agent_labels"].cpu().numpy().tolist()
    vehicle_bboxes = [box for box, label in zip(vehicle_bboxes, vehicle_bboxes_lb) if label]

    # --- Target pedestrian agent boxes ---
    pedestrian_bboxes = targets["pedestrian_agent_states"].cpu().numpy().tolist()
    pedestrian_bboxes_lb = targets["pedestrian_agent_labels"].cpu().numpy().tolist()
    pedestrian_bboxes = [box for box, label in zip(pedestrian_bboxes, pedestrian_bboxes_lb) if label]

    if targets["traffic_light_state"].cpu().int().numpy()[0] == 0:
        traffic_light = "GREEN"
    else:
        traffic_light = "RED"

    # --- Inference ---
    logger.info("Running inference")
    feat_copy = {k: v.unsqueeze(0) for k, v in features.items()}
    agent.eval()

    start = time.perf_counter()
    with torch.no_grad():
        outputs = agent.forward(feat_copy, gaze_flag=use_gaze, neighbours_flag=use_neighbors)
    end = time.perf_counter()

    inference_time = end - start  # seconds
    fps = 1 / inference_time  # frames per second
    ms = inference_time * 1000  # milliseconds
    logger.info(f"Inference complete: {fps:.1f} FPS | {ms:.1f} ms")
    ego_trajectory = outputs['trajectory'].squeeze(0).detach().cpu().tolist()

    # --- Predicted vehicle agent boxes ---
    pred_vehicle_bboxes = outputs["vehicle_agent_states"].cpu()[0].numpy().tolist()
    pred_vehicle_bboxes_lb = outputs["vehicle_agent_labels"].cpu()[0].numpy().tolist()
    pred_vehicle_bboxes = [box for box, label in zip(pred_vehicle_bboxes, pred_vehicle_bboxes_lb) if label]

    # --- Predicted pedestrian agent boxes ---
    pred_pedestrian_bboxes = outputs["pedestrian_agent_states"].cpu()[0].numpy().tolist()
    pred_pedestrian_bboxes_lb = outputs["pedestrian_agent_labels"].cpu()[0].numpy().tolist()
    pred_pedestrian_bboxes = [box for box, label in zip(pred_pedestrian_bboxes, pred_pedestrian_bboxes_lb) if label]

    if outputs["traffic_light_state"].cpu()[0].sigmoid().round().int().numpy()[0] == 0:
        pred_traffic_light = "GREEN"
    else:
        pred_traffic_light = "RED"

    with torch.no_grad():
        feat_copy = {k: v.unsqueeze(0) for k, v in features.items()}
        outputs = agent.forward(feat_copy, gaze_flag=False, neighbours_flag=False)
    ego_trajectory_no_unreliables = outputs['trajectory'].squeeze(0).detach().cpu().tolist()

    semantic_map = img_to_base64(draw_semantic(targets['bev_semantic_map']))
    pred_semantic_map = img_to_base64(draw_semantic(outputs['bev_semantic_map'][0].argmax(dim=0)))

    stop_lines = targets["stop_lines"]
    if stop_lines:
        stop_lines = targets["stop_lines"][0].squeeze(1).detach().cpu().tolist()

    # --- Package data ---
    data = {
        "image": f"data:image/png;base64,{encoded_img}",
        "gaze_image":f"data:image/png;base64,{gaze_img_encoded}",
        "trajectories": trajectories_tensor,
        "ego_trajectory": ego_trajectory,
        "ego_trajectory_no_unreliables": ego_trajectory_no_unreliables,
        "true_trajectory": true_trajectory,
        "vehicle_bboxes": vehicle_bboxes,
        "pedestrian_bboxes": pedestrian_bboxes,
        "semantic": f"data:image/png;base64,{semantic_map}",
        "pred_semantic": f"data:image/png;base64,{pred_semantic_map}",
        "light": traffic_light,
        "stop_line": stop_lines,
        "pred_vehicle_bboxes": pred_vehicle_bboxes,
        "pred_pedestrian_bboxes": pred_pedestrian_bboxes,
        "pred_light": pred_traffic_light,
        "lidar_raw": lidar_list,
        "fps": fps,
        "ms": ms
    }

    json_data = json.dumps(data).encode('utf-8')
    gzip_data = gzip.compress(json_data, compresslevel=5)
    logger.info("Returning scenario_data")
    return Response(gzip_data, mimetype='application/json', headers={'Content-Encoding': 'gzip'})

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    global scene_loader
    global agent

    logger.info(f"Creating agent")
    agent = instantiate(cfg.agent)

    # Build scene loader with no caching
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)

    logger.info(f"Creating scene loader")
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    logger.info(f"Found {len(scene_loader)} raw scenarios")

    # Start flask server
    app.run(debug=True)


if __name__ == "__main__":
    main()
