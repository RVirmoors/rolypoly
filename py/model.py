"""
Rolypoly timing model
2023 rvirmoors
"""

import torch
import torch.nn as nn

class Basic(nn.Module):
    def __init__(self, in_channels=13, out_channels=10):
        super(Basic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return x[:self.out_channels]