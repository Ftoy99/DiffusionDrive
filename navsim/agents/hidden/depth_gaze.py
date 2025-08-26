import torch  # need this import for transformers
from transformers import pipeline, AutoImageProcessor

print("Loading LiheYoung/depth-anything-small-hf")
# load pipe
processor = AutoImageProcessor.from_pretrained(
    "LiheYoung/depth-anything-small-hf", use_fast=True
)

pipe = pipeline(
    task="depth-estimation",
    model="LiheYoung/depth-anything-small-hf",
    image_processor=processor,
    device=0
)

def depth_inf(image):
    # inference
    depth = pipe(image)["depth"]
    return depth
