from transformers import pipeline


print("Loading LiheYoung/depth-anything-small-hf")
# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")


def depth_inf(image):
    # inference
    depth = pipe(image)["depth"]
    return depth
