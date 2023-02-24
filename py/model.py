"""
Rolypoly timing model
2023 rvirmoors

Very much inspired by A Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import data
import time

# === TEST / TOY MODELS ===

class Basic(nn.Module):
    def __init__(self, in_channels=14, out_channels=14):
        # in: 14 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau_d, tau_g)
        # out: 14 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau_d, tau_g)
        super(Basic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return x + 0.66

class Swing(nn.Module):
    def __init__(self, in_channels=14, out_channels=14):
        # in: 14 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau_d, tau_g)
        # out: 14 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau_d, tau_g)
        super(Swing, self).__init__()
        torch.set_printoptions(precision=2, sci_mode=False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, _, x):
        for i in range(x.shape[-1]):
            if data.upbeat(x[0, 11, i]):
                # if we're on an upbeat, nudge the note forward
                nudge = data.bartime_to_ms(0.05, x[0, :, i])
                x[0, 12, i] = nudge
        return x

# === HELPER CLASSES FOR TRANSFORMER ===

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False 
    - from https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """ input: x input (batch, seq_len, channels)
        output: y projection (batch, seq_len, channels)
    - adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py"""





# === TRANSFORMER CLASS ===

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_enc, x_dec):
        return x_dec + 0.01

        



# === TESTS ===
if __name__ == '__main__':
    test = torch.tensor([[[42, 36, 38, 42, 36],
                          [70, 60, 111, 105, 101],
                          [120, 120, 140, 140, 140],
                          [1, 1, 1, 1.5, 1.5],
                          [0, 0.5, 0, 0.33, 0.66]]])
    #print(data.readScore(test).shape)
    #print(readScore(test)[:, :10, :])
    x_enc = data.readScore(test)
    x_dec = torch.zeros(1, 14, 0)
    x_dec = data.readScoreLive(test[:,:,:2], x_dec)
    print(x_dec[:,:,0])
    #feat = x.squeeze(0)
    
    s = Transformer()
    start = time.time()
    print(s(x_enc, x_dec)[:,:,0])
    print(time.time() - start, "s")