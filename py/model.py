"""
Rolypoly timing model
2023 rvirmoors
"""

import torch
import torch.nn as nn

class Basic(nn.Module):
    def __init__(self, in_channels=13, out_channels=10):
        # in: 13 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau)
        # out: 10 channels (tau, 9 drum velocities)
        super(Basic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        tall_tau = 0.1 * torch.ones(1, x.shape[1])
        y = torch.cat((tall_tau, x[:9]), dim=0)
        return y