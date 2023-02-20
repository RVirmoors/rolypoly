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
        self.register_buffer('x_enc', torch.zeros(1, 12, 0))
        self.register_buffer('x_dec', torch.zeros(1, 14, 0)) # actuals
        self.register_buffer('y_hat', torch.zeros(1, 14, 0)) # predictions

        # REGISTER METHODS
        self.register_method(
            'forward',
            in_channels = 5, # hit, vel, tempo, tsig, pos_in_bar
            in_ratio = 1,
            out_channels = 14, # 9 drum velocities, bpm, tsig, pos_in_bar, tau_drums, tau_guitar
            out_ratio = 1,
            input_labels = ['hit', 'vel', 'bpm', 'tsig', 'pos_in_bar'],
            output_labels = ['K', 'S', 'HI-c', 'HI-o', 'T-l', 'T-m', 'T-h', "cr", 'rd', 'bpm', 'tsig', 'pos_in_bar', 'tau', 'tau_g'],
            test_buffer_size = 512,
            test_method = False
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
            self.x_enc = data.readScore(input)
            out = torch.cat((self.x_enc, torch.zeros(1, 2, self.x_enc.shape[2])), dim=1)
            return out

        if self.play[0]:
            #return torch.zeros(1, 14, 1)
            if m_buf_size == 1 and input[:, 0, 0] == 666: # just one onset
                print("one onset")
                if self.x_dec.shape[2] == 0:
                    return torch.zeros(1, 14, 1) # can't modify x_dec yet!
                # update x_dec[:,:,14] with realised tau_guitar
                self.x_dec = data.readLiveOnset(input, self.x_dec)
                return self.x_dec
            else: # full buffer = receiving drum hits
                print("full buffer")
                # add latest hits to x_dec
                before = self.x_dec.shape[-1]
                self.x_dec = data.readScoreLive(input, self.x_dec)
                latest = self.x_dec.shape[-1] - before
                # get predictions
                preds = self.pretrained(self.x_enc, self.x_dec)
                # preds[:, -2, -1] = before # set tau_drums to latest             
                # update y_hat with latest predictions
                self.y_hat = torch.cat((self.y_hat, preds[:,:,-latest:]), dim=-1) 
                return self.y_hat[:, :, -latest:]
        else:
            out = torch.cat((self.x_enc, torch.zeros(1, 2, self.x_enc.shape[2])), dim=1)
            return out

if __name__ == '__main__':
    test = False

    pretrained = model.Swing()
    pretrained.eval()
    m = ExportRoly(pretrained=pretrained)
    if not test:
        m.export_to_ts('../help/roly.ts')
        print("exported to ../help/roly.ts")
    else:
        score = torch.tensor([[[42, 36, 38, 42, 36],
                          [70, 60, 111, 105, 101],
                          [120, 120, 140, 140, 140],
                          [1, 1, 1.5, 1.5, 1.5],
                          [0, 0.5, 0.33, 0.33, 0.66]]])
        live_drums = torch.tensor([[[42, 36],
                            [70, 60],
                            [120, 120],
                            [1, 1],
                            [0, 0.5]]])
        guit = torch.tensor([[[666],[0.6],[120],[1],[0]]])
        m.set_read(True)
        out = m.forward(score)
        print("read -> enc: ", out[:,:,-1], out.shape)
        m.set_read(False)
        print("=====================")
        m.set_play(True)
        #out = m.forward(live_drums)
        #print("drums -> y_hat: ", out[:,:,-1], out.shape)
        out = m.forward(live_drums)
        print("drums2 > y_hat: ", out, out.shape)
        out = m.forward(guit)
        print("guit -> dec: ", out, out.shape)