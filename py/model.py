"""
Rolypoly timing model
2023 rvirmoors
"""

import torch
import torch.nn as nn
import data
import time

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
    def __init__(self, in_channels=14, out_channels=14):
        # in: 14 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau_d, tau_g)
        # out: 14 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau_d, tau_g)
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transformer_model = nn.Transformer(d_model=14, nhead=14)

    def forward(self, x_enc, x_dec):
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        x_enc = torch.cat((x_enc, torch.zeros(1, 2, x_enc.shape[2])), dim=1)
        x_enc = x_enc.permute(2, 0, 1)
        x_dec = x_dec.permute(2, 0, 1)
        y = self.transformer_model(x_enc, x_dec)
        y = y.permute(1, 2, 0)
        return y
        



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