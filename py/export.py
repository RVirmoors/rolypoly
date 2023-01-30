## nn~ module that receives parsed drum info and plays the drum track
# 2023 rvirmoors

import torch, torch.nn as nn
import nn_tilde
from typing import List, Tuple

import data # data helper methods
from constants import ROLAND_DRUM_PITCH_CLASSES

class ExportRoly(nn_tilde.Module):

    def __init__(self):
        super().__init__()
        # REGISTER ATTRIBUTES
        # read: load a new drum track
        self.register_attribute('read', False)
        # play: receive X(t), generate Y(t) and send to host
        self.register_attribute('play', False)

        # REGISTER BUFFERS
        self.register_buffer('X', torch.zeros(1, 10, 8192), persistent=False)
        self.register_buffer('t', torch.zeros(1))

        # REGISTER METHODS
        self.register_method(
            'forward',
            in_channels = 5, # hit, vel, tempo, tsig, pos_in_bar
            in_ratio = 1,
            out_channels = 10, # tau(delay) + 9 drum velocities
            out_ratio = 1,
            input_labels = ['hit', 'vel', 'tempo', 'tsig', 'pos_in_bar'],
            output_labels = ['tau', 'K', 'S', 'HI-c', 'HI-o', 'T-l', 'T-m', 'T-h', "cr", 'rd']
        )

    # defining attribute getters
    # WARNING : typing the function's ouput is mandatory
    @torch.jit.export
    def get_play(self) -> bool:
        return self.play[0]

    @torch.jit.export
    def get_read(self) -> bool:
        return self.read[0]

    # defining attribute setters
    @torch.jit.export
    def set_play(self, value: bool):
        self.play = (value,)
        return 0

    @torch.jit.export
    def set_read(self, value: bool):
        self.read = (value,)
        return 0

    # definition of the main method
    @torch.jit.export
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.read[0]:
            # stack the input onto X
            #self.X = torch.cat((self.X, input), dim=2)
            self.X = torch.cat((input, input), dim=1)
            self.set_read(False)
            return self.X

        if self.play[0]:
            self.t += 1
            return torch.cat((input, input), dim=1) * self.t[0]
        else:
            # play X
            return self.X

if __name__ == '__main__':
    model = ExportRoly()
    model.export_to_ts('../help/roly.ts')