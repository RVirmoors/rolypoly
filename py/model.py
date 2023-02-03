"""
Rolypoly timing model
2023 rvirmoors
"""

import torch
import torch.nn as nn

class Basic(nn.Module):
    def __init__(self, in_channels=13, out_channels=13):
        # in: 13 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau)
        # out: 13 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau)
        super(Basic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return x + 0.66