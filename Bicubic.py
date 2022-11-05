import torch
import torch.nn as nn


class BICUBIC(nn.Module):
    def __init__(self):
        super(BICUBIC, self).__init__()
        self.up = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

    def forward(self, x):
        return self.up(x)
