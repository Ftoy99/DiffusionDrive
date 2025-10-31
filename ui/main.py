import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import hydra
from PIL import Image
from hydra.utils import instantiate
from omegaconf import DictConfig
from flask import Flask, render_template, request, send_file, Response, jsonify

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.hidden_v2.hidden_config import HiddenConfig
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from test.visualization import draw_bev, draw_semantic
import os

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scenarios")
def scenarios():
    if scene_loader is None:
        return '<option value="">No scenarios available</option>'

    return "".join(
        f'<option value="{token}">{token}</option>'
        for token in scene_loader.tokens
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

@app.route("/lidar")
def lidar():
    img_tag = f'<img src="/lidar_png?_={time.time()}" class="w-full h-32 object-contain">'
    return img_tag

@app.route("/lidar_png")
def lidar_png():
    global features
    if features is None:
        return "<div>No features initialized</div>"

    img_tensor = features["lidar_feature"]
    img_array = draw_bev(img_tensor,targets["trajectory"])

    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

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

@app.route("/run_inference", methods=["POST"])
def run_inference():
    global scene_loader
    global agent
    global features
    global targets
    global outputs
    feat_copy = features
    model = request.form.get("model")
    if agent.checkpoint_path is not model:
        logger.info(f"Loading from pretrained")
        agent.checkpoint_path = model
        agent.initialize()
        agent.eval()

    feat_copy = {k: v.unsqueeze(0) for k, v in feat_copy.items()} # Add batch dim
    outputs = agent.forward(feat_copy)
    return render_template("inference_results.html")

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


@app.route("/scenario_data", methods=["GET"])
def scenario_data():
    logger.info("Get scenario_data")
    global features, targets, agent_input, agent, outputs
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
        (lidar_pc[1, :] >= config.lidar_min_y) & (lidar_pc[1, :] <= config.lidar_max_y)
    )
    lidar_pc = lidar_pc[:, mask]
    lidar_list = lidar_pc.T.tolist()

    # --- Ground truth trajectories ---
    trajectories_tensor = features["trajectories"].cpu().numpy().tolist()
    true_trajectory = targets["trajectory"].cpu().tolist()

    # --- Filtered agent boxes ---
    bboxes = targets["agent_states"].cpu().numpy().tolist()
    bboxes_lb = targets["agent_labels"].cpu().numpy().tolist()
    bboxes = [box for box, label in zip(bboxes, bboxes_lb) if label]

    # --- Inference ---
    logger.info("Running inference")
    feat_copy = {k: v.unsqueeze(0) for k, v in features.items()}
    agent.eval()

    start = time.perf_counter()
    outputs = agent.forward(feat_copy)
    end = time.perf_counter()

    inference_time = end - start  # seconds
    fps = 1 / inference_time  # frames per second
    ms = inference_time * 1000  # milliseconds

    logger.info(f"Inference complete: {fps:.1f} FPS | {ms:.1f} ms")

    ego_trajectory = outputs['trajectory'].squeeze(0).detach().cpu().tolist()
    # --- Package data ---
    data = {
        "image": f"data:image/png;base64,{encoded_img}",
        "gaze_image":f"data:image/png;base64,{gaze_img_encoded}",
        "trajectories": trajectories_tensor,
        "ego_trajectory": ego_trajectory,
        "true_trajectory": true_trajectory,
        "bboxes": bboxes,
        "lidar_raw": lidar_list
    }

    logger.info("Returning scenario_data")
    return jsonify(data)

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
