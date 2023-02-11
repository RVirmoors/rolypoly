"""
Rolypoly timing model
2023 rvirmoors
"""

import torch
import torch.nn as nn
import data
import time

class Basic(nn.Module):
    def __init__(self, in_channels=13, out_channels=13):
        # in: 13 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau)
        # out: 13 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau)
        super(Basic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return x + 0.66

class Swing(nn.Module):
    def __init__(self, in_channels=13, out_channels=13):
        # in: 13 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau)
        # out: 13 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau)
        super(Swing, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        for i in range(x.shape[-1]):
            if data.upbeat(x[0, 11, i]):
                # if we're on an upbeat, nudge the note forward
                nudge = data.bartime_to_ms(0.05, x[0, :, i])
                x[0, 12, i] = nudge
        return x + 0.01

class Transformer(nn.Module):
    def __init__(self, in_channels=13, out_channels=13):
        # in: 13 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau)
        # out: 13 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau)
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transformer_model = nn.Transformer(d_model=13, nhead=13)

    def forward(self, x):
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        x = x.permute(2, 0, 1)
        x = self.transformer_model(x, x)
        x = x.permute(1, 2, 0)
        return x
        



# === TESTS ===
if __name__ == '__main__':
    test = torch.tensor([[[42, 36, 38, 42, 36],
                          [70, 60, 111, 105, 101],
                          [120, 120, 140, 140, 140],
                          [1, 1, 1, 1.5, 1.5],
                          [0, 0.5, 0, 0.33, 0.66]]])
    print(data.readScore(test).shape)
    #print(readScore(test)[:, :10, :])
    x = data.readScore(test)
    #feat = x.squeeze(0)
    
    s = Swing()
    # warmup pass
    x = torch.zeros(1, 13, 5)
    s(x)
    start = time.time()
    print(s(x))
    print(time.time() - start, "s")