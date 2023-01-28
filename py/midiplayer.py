## nn~ module that receives parsed drum info and plays the drum track
# 2023 rvirmoors

import torch, torch.nn as nn
import nn_tilde
from typing import List, Tuple

import data # data helper methods
from constants import ROLAND_DRUM_PITCH_CLASSES

class MidiPlayer(nn_tilde.Module):

    def __init__(self):
        super().__init__()
        # REGISTER ATTRIBUTES
        self.register_attribute('play', False)
        self.register_attribute('read', True)

        # REGISTER BUFFERS
        self.register_buffer('X', torch.zeros(1, 10, 1000))
        self.register_buffer('time', torch.zeros(1))

        # REGISTER METHODS
        self.register_method(
            'forward',
            in_channels = 10, # timestamp + 9 drum channels
            in_ratio = 1,
            out_channels = 10, # time delay + 9 drum channels
            out_ratio = 1,
            input_labels = ['timestamp', 'K', 'S', 'HI-c', 'HI-o', 'T-l', 'T-m', 'T-h', "cr", 'rd'],
            output_labels = ['delay', 'K', 'S', 'HI-c', 'HI-o', 'T-l', 'T-m', 'T-h', "cr", 'rd']
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
            # record the input
            self.X = input
            self.set_read(False)
            return self.X

        if self.play[0]:
            # parse X, where the first column is the timestamp
            # and the other columns are the drum channels

            self.time += 1
            return input * self.time[0]
        else:
            # record the input
            return self.X

if __name__ == '__main__':
    # create a new instance of the model
    model = MidiPlayer()
    # save the module
    model.export_to_ts('../help/midiplayer.ts')