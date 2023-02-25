"""
Rolypoly Python implementation
2023 rvirmoors

export the model to a .ts file, loadable by the rolypoly~ Max object
"""

import torch, torch.nn as nn
import nn_tilde

import data # data helper methods
import model
torch.set_printoptions(sci_mode=False)

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
        # finetune: train the model based on the just ended performance
        self.register_attribute('finetune', False)

        # REGISTER BUFFERS
        self.register_buffer('x_enc', torch.zeros(1, 0, 12)) # score
        self.register_buffer('x_dec', torch.randn(1, 1, 14)) # actuals
        self.register_buffer('y_hat', torch.zeros(1, 0, 14)) # predictions

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

    @torch.jit.export
    def set_finetune(self, value: bool):
        self.finetune = (value,)
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
            self.x_dec = self.x_enc[0, 0, :].unsqueeze(0).unsqueeze(0).clone().detach()
            self.x_dec = torch.cat((self.x_dec, torch.zeros(1, 1, 2)), dim=2)
            self.y_hat = torch.zeros(1, 0, 14)
            return out

        if self.play[0]:
            if m_buf_size == 1 and input[0, 0, 0] == 666: # just one onset
                print("one onset")
                if self.x_dec.shape[1] <= 1:
                    return torch.zeros(1, 1, 14) # can't modify x_dec yet!
                # update x_dec[:,:,14] with realised tau_guitar
                self.x_dec = data.readLiveOnset(input, self.x_dec, self.x_enc)
                return self.x_dec
            else: # receiving drum hits
                next_notes = data.readScoreLive(input)
                num_samples = next_notes.shape[1]
                # get predictions
                xe = self.x_enc.clone().detach()
                xd = self.x_dec.clone().detach()
                data.dataScaleDown(xe)
                data.dataScaleDown(xd)
                self.x_dec = self.pretrained.generate(xe, xd, num_samples)
                data.dataScaleUp(self.x_dec)
                # update y_hat and x_dec with latest predictions
                self.y_hat = torch.cat((self.y_hat, self.x_dec[:, -num_samples:, :]), dim=1)
                # reset x_dec[13] to 0, waiting for live tau_guitar
                self.x_dec[:, -num_samples:, 13] = 0
                # return predictions
                return self.y_hat[:, -num_samples:, :]

        elif self.finetune[0]:
            y_hat = self.y_hat.clone().detach()
            data.dataScaleUp(y_hat)

            if input[0,0,0] == 0:
                return self.x_dec
            return y_hat

        else:
            out = torch.cat((self.x_enc, torch.zeros(1, self.x_enc.shape[1], 2)), dim=2)
            return out

if __name__ == '__main__':
    test = False
    pretrain = True

    if pretrain:
        checkpoint = torch.load('out/ckpt.pt')
        config = checkpoint['config']
        pretrained = model.Transformer(config)
        pretrained.load_state_dict(torch.load('out/model_best.pt'))
        print("Loaded pretrained model:", checkpoint['iter_num'], "epochs, loss:", checkpoint['best_val_loss'].item())
    else:
        config = model.Config()
        pretrained = model.Transformer(config)

    #pretrained = model.Basic()
    pretrained.eval()
    m = ExportRoly(pretrained=pretrained)
    if not test:
        m.export_to_ts('../help/roly.ts') # TODO: make this a command line argument
        print("Exported model to ../help/roly.ts")
    else:
        score = torch.tensor([[[42, 70, 120, 1, 0],
                            [36, 60, 120, 1, 0.5],
                            [38, 111, 140, 1.5, 0.33],
                            [42, 105, 140, 1.5, 0.33],
                            [36, 101, 140, 1.5, 0.66]]])
        live_drums = torch.tensor([[[42, 70, 120, 1, 0],
                            [36, 60, 120, 1, 0.5]]])
        guit = torch.tensor([[[666, 0.666, 120, 1, 0]]])

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