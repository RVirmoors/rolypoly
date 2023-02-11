"""
Rolypoly Python implementation
2023 rvirmoors

export the model to a .ts file, loadable by the rolypoly~ Max object
"""

import torch, torch.nn as nn
import nn_tilde
from typing import List, Tuple

import data # data helper methods
import model

class ExportRoly(nn_tilde.Module):

    def __init__(self, pretrained):
        super().__init__()
        self.pretrained = pretrained

        # REGISTER ATTRIBUTES
        # read: load a new drum track
        self.register_attribute('read', False)
        # play: receive X(t), generate Y(t) and send to host
        self.register_attribute('play', True)
        # generate: trigger notes irrespective of score
        self.register_attribute('generate', False)

        # REGISTER BUFFERS
        self.register_buffer('X', torch.ones(1, 13, 512))

        # REGISTER METHODS
        self.register_method(
            'forward',
            in_channels = 5, # hit, vel, tempo, tsig, pos_in_bar
            in_ratio = 1,
            out_channels = 13, # 9 drum velocities, bpm, tsig, pos_in_bar, tau
            out_ratio = 1,
            input_labels = ['hit', 'vel', 'bpm', 'tsig', 'pos_in_bar'],
            output_labels = ['K', 'S', 'HI-c', 'HI-o', 'T-l', 'T-m', 'T-h', "cr", 'rd', 'bpm', 'tsig', 'pos_in_bar', 'tau'],
            test_buffer_size = 512
        )

    # defining attribute getters
    # WARNING : typing the function's ouput is mandatory
    @torch.jit.export
    def get_play(self) -> bool:
        return self.play[0]

    @torch.jit.export
    def get_read(self) -> bool:
        return self.read[0]

    @torch.jit.export
    def get_generate(self) -> bool:
        return self.generate[0]

    # defining attribute setters
    @torch.jit.export
    def set_play(self, value: bool):
        self.play = (value,)
        return 0

    @torch.jit.export
    def set_read(self, value: bool):
        self.read = (value,)
        return 0

    @torch.jit.export
    def set_generate(self, value: bool):
        self.generate = (value,)
        return 0

    # definition of the main method
    @torch.jit.export
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        m_buf_size = input.shape[-1]

        if self.read[0]:
            self.X = data.readScore(input)
            return self.X

        if self.play[0]:
            x = data.readScore(input)
            y = self.pretrained(x)
            # output is a single timestep
            return y
        else:       
            return self.X

if __name__ == '__main__':
    pretrained = model.Transformer()
    pretrained.eval()
    # warmup pass
    x = torch.zeros(1, 13, 5)
    pretrained(x)
    m = ExportRoly(pretrained=pretrained)
    m.export_to_ts('../help/roly.ts')