"""
Rolypoly Python implementation
2023 rvirmoors

export the model to a .ts file, loadable by the rolypoly~ Max object
"""

import torch, torch.nn as nn
import nn_tilde

import data # data helper methods
import train_gmd # for testing GMD
import finetune
import constants
import model
from typing import List, Tuple
torch.set_printoptions(sci_mode=False, linewidth=200, precision=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# ============== MAIN EXPORT CLASS ==============

class ExportRoly(nn_tilde.Module):

    def __init__(self, pretrained, params: List[torch.Tensor]):
        super().__init__()
        self.pretrained = pretrained
        self.params = params

        # REGISTER ATTRIBUTES
        # read: load a new drum track
        self.register_attribute('read', False)
        # play: receive X(t), generate Y(t) and send to host
        self.register_attribute('play', True)
        # generate: trigger notes irrespective of score
        self.register_attribute('generate', False)
        # finetune: train the model based on the just ended performance
        self.register_attribute('finetune', False)
        # score_filter: play only notes that appear in the score (x_enc)
        self.register_attribute('score_filter', True)

        # REGISTER BUFFERS
        self.register_buffer('x_enc', torch.zeros(1, 0, 12)) # score
        self.register_buffer('x_dec', torch.randn(1, 1, 14)) # actuals
        self.register_buffer('y_hat', torch.zeros(1, 0, 14)) # predictions
        self.register_buffer('diag', torch.zeros(1, 1, 14)) # diagnostics (finetune)

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

    @torch.jit.export
    def get_finetune(self) -> bool:
        return self.finetune[0]

    @torch.jit.export
    def get_score_filter(self) -> bool:
        return self.score_filter[0]

    # defining attribute setters
    @torch.jit.export
    def set_play(self, value: bool):
        if value: # if play is set to true, reset x_dec and y_hat
            self.x_dec = torch.zeros(1, 1, 14)
            self.y_hat = torch.zeros(1, 0, 14)
        self.play = (value,)
        return 0

    @torch.jit.export
    def set_read(self, value: bool):
        if value: # if read is set to true, reset x_enc, x_dec and y_hat
            self.x_enc = torch.zeros(1, 0, 12)
            self.x_dec = torch.zeros(1, 1, 14)
            self.y_hat = torch.zeros(1, 0, 14)
        self.read = (value,)
        return 0

    @torch.jit.export
    def set_generate(self, value: bool):
        self.generate = (value,)
        return 0

    @torch.jit.export
    def set_finetune(self, value: bool):
        self.finetune = (value,)
        return 0

    @torch.jit.export
    def set_score_filter(self, value: bool):
        self.score_filter = (value,)
        return 0

    # definition of the main method
    @torch.jit.export
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: tensor (batch, m_buf_size, 5 chans)
        # output: tensor (batch, time, 14 chans)
        m_buf_size = input.shape[1]

        if self.read[0]:
            self.x_enc = data.readScore(input)
            out = torch.cat((self.x_enc, torch.zeros(1, self.x_enc.shape[1], 2)), dim=2)
            # initialise x_dec and y_hat
            self.x_dec = torch.zeros(1, 1, 14)
            self.y_hat = torch.zeros(1, 0, 14)
            return out

        if self.play[0]:
            if m_buf_size == 1 and input[0, 0, 0] == 666: # just one onset
                print("one onset")
                if self.x_dec.shape[1] <= 1 or self.x_dec.shape[1] > self.x_enc.shape[1] - 2:
                    return torch.zeros(1, 1, constants.X_DECODER_CHANNELS) # can't modify x_dec yet!
                # update x_dec[:,:,14] with realised tau_guitar
                self.x_dec = data.readLiveOnset(input, self.x_dec, self.x_enc)
                return self.x_dec
            else: # receiving drum hits
                next_notes = data.readScoreLive(input)
                num_samples = next_notes.shape[1]
                # get predictions
                if self.x_enc.shape[1]:
                    xe = self.x_enc.clone().detach()
                else:
                    # x_enc hasn't been loaded yet (warmup)
                    xe = torch.zeros(1, 10, constants.X_ENCODER_CHANNELS)
                if self.x_dec.shape[1] == 1:
                    xd = xe[0,0].clone().detach().unsqueeze(0).unsqueeze(0)
                    xd = torch.cat((xd, torch.zeros(1, 1, 2)), dim=2)
                else:
                    xd = self.x_dec.clone().detach()
                data.dataScaleDown(xe)
                data.dataScaleDown(xd)
                self.x_dec = self.pretrained.generate(xe, xd, num_samples)
                data.dataScaleUp(self.x_dec)
                # update y_hat and x_dec with latest predictions
                self.y_hat = torch.cat((self.y_hat, self.x_dec[:, -num_samples:]), dim=1)
                if self.score_filter[0] and self.x_enc.shape[1]:
                    # set non x_enc notes to zero
                    self.y_hat[:,:self.x_enc.shape[1],:constants.X_ENCODER_CHANNELS][self.x_enc[:,:self.y_hat.shape[1]] == 0] = 0
                    self.x_dec[:,:self.x_enc.shape[1],:constants.X_ENCODER_CHANNELS][self.x_enc[:,:self.x_dec.shape[1]] == 0] = 0
                # reset x_dec[13] to 0, waiting for live tau_guitar
                self.x_dec[:, -num_samples:, constants.INX_TAU_G] = 0
                # return predictions
                out = self.y_hat[:, -num_samples:, :].clone().detach()
                out[:, :, 12:] = data.bartime_to_ms(out[:, :, 12:], out)
                return out

        elif self.finetune[0]:
            if input[0, 0, 0] == 1: # just one onset
                follow = input[0,0,1]
                self.pretrained, self.params, loss, self.diag = finetune.finetune(self.pretrained, self.params, self.x_enc, self.x_dec, self.y_hat, Follow=follow)
                return loss
            else: # diagnostics
                return self.diag

        else:
            out = torch.cat((self.x_enc, torch.zeros(1, self.x_enc.shape[1], 2)), dim=2)
            return out

# ==================== TESTS ====================

def test_toy(m):
    score = torch.tensor([[[42, 70, 120, 1, 0],
                        [36, 60, 120, 1, 0.5],
                        [38, 111, 140, 1.5, 1.33],
                        [42, 105, 140, 1.5, 1.33],
                        [36, 101, 140, 1.5, 1.66]]])
    live_drums = torch.tensor([[[42, 70, 120, 1, 0],
                        [36, 60, 120, 1, 0.5]]])
    guit = torch.tensor([[[666, 66.6, 120, 1, 0]]])

    m.set_read(True)
    out = m.forward(score)
    print("read -> enc: ", out[:,-1,:], out.shape)
    m.set_read(False)
    print("=====================")
    m.set_play(True)
    out = m.forward(live_drums)
    print("drums2 > y_hat: ", out, out.shape)
    out = m.forward(guit)
    print("guit -> dec: ", out, out.shape)

def test_gmd(m):
    x_take = data.loadX_takeFromCSV('gmd.csv')
    xd, xe, y = train_gmd.getTrainDataFromX_take(x_take)
    x_dec = xd[0].unsqueeze(0).unsqueeze(0) #torch.cat((xe[0, :].unsqueeze(0).unsqueeze(0).clone().detach(), torch.zeros(1, 1, 2)), dim=2)
    x_enc = xe.unsqueeze(0)
    y = y.unsqueeze(0)
    print("first x_dec:\n", x_dec[0, :3], x_dec.shape)
    print("first x_enc:\n", x_enc[0, :3], x_enc.shape)

    for i in range(24):#x_enc.shape[1]):
        xd = x_dec.clone().detach()
        xe = x_enc.clone().detach()
        data.dataScaleDown(xd)
        data.dataScaleDown(xe)
        x_dec = m.pretrained.generate(xe, xd, 1)
        data.dataScaleUp(x_dec)
        x_dec[:, -1:, 13] = 0
        # set non x_enc notes to zero
        x_dec[:,:,:x_enc.shape[2]][x_enc[:,:x_dec.shape[1]] == 0] = 0

        xd = x_dec.clone().detach()
        print("x_dec final out:\n", xd[:,-1], xd.shape)
        print("y:\n", y[:,i], y.shape)
        xd[:, :, 12:14] = data.bartime_to_ms(xd[:, :, 12:14], xd)

    guit = torch.tensor([[[666, 66.6, 84, 1, 1.56]]])
    out = data.readLiveOnset(guit, x_dec, x_enc)
    #  print("guit -> dec: ", out[:,-8:], out.shape)

# ==================== MAIN =====================

if __name__ == '__main__':
    test = False
    pretrain = True
    basic = False

    if basic:
        pretrained = model.Basic()
    else:
        if pretrain:
            checkpoint = torch.load('out/ckpt.pt', map_location=device)
            config = checkpoint['config']
            pretrained = model.Transformer(config)
            pretrained.load_state_dict(torch.load('out/model_best.pt', map_location=device))
            print("Loaded pretrained model:", checkpoint['iter_num'], "epochs, loss:", checkpoint['best_val_loss'].item())
        else:
            config = model.Config()
            pretrained = model.Transformer(config)

    # pretrained = torch.jit.script(pretrained)
    m = ExportRoly(pretrained=pretrained, params=list(pretrained.parameters()))
    
    if not test:
        m.export_to_ts('../help/roly.ts') # TODO: make this a command line argument
        print("Exported model to ../help/roly.ts")
    else:
        # test_toy(m)
        test_gmd(m)

