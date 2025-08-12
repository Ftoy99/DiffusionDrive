import torch
import torch.nn as nn

layers = nn.Sequential(
    nn.Conv2d(64, 256, 1),   # for 64x72x72
    nn.Conv2d(64, 256, 1),   # for 64x36x36
    nn.Conv2d(128, 256, 1),  # for 128x18x18
    nn.Conv2d(256, 256, 1),  # for 256x9x9
    nn.Conv2d(512, 256, 1),  # for 512x5x5
)

# Fake tensors for each layer input:
tensors = [
    torch.randn(1, 64, 72, 72),
    torch.randn(1, 64, 36, 36),
    torch.randn(1, 128, 18, 18),
    torch.randn(1, 256, 9, 9),
    torch.randn(1, 512, 5, 5),
]

# Applying each conv layer on its corresponding tensor:
outputs = [layer(t) for layer, t in zip(layers, tensors)]

tokenized = [x.flatten(2).transpose(1, 2) for x in outputs]

for t in tokenized:
    print(t.shape)  # (Batch, Tokens, 256)

