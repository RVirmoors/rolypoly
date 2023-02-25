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
import math
import inspect
from dataclasses import dataclass

# === TEST / TOY MODELS ===

class Basic(nn.Module):
    def __init__(self, in_channels=14, out_channels=14):
        # in: 14 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau_d, tau_g)
        # out: 14 channels (9 drum velocities, bpm, tsig, pos_in_bar, tau_d, tau_g)
        super(Basic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, _, x):
        return x + 1/127

    def generate(self, _, x, num_samples:int=1):
        out = x
        for steps in range(num_samples):
            x = self.forward(_, x)
            x = x[:, -1:, :]
            out = torch.cat((out, x), dim=1)
        return out

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

# === TRANSFORMER HYPERPARAMETERS ===
@dataclass
class Config:
    n_layers = 4 # number of block layers
    block_size = 16 # number of hits in a block
    dropout = 0.1

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

class SelfAttention(nn.Module):
    """ input: x_dec/x_enc input (batch, seq_len, channels)
        output: y projection (batch, seq_len, channels)
    - adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, config, causal=True) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.block_size = config.block_size
        self.causal = causal
        if self.causal:
            self.n_chans = data.X_DECODER_CHANNELS      # 14
            self.n_heads = data.X_DECODER_CHANNELS // 2 # 7
            # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
            if not self.flash:
                #print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
                # causal mask to ensure that attention is only applied to the left in the input sequence
                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))
        else:
            self.n_chans = data.X_ENCODER_CHANNELS      # 12
            self.n_heads = data.X_ENCODER_CHANNELS // 3 # 4
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_chans, 3 * self.n_chans, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_chans, self.n_chans, bias=False)
        # regularization
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)
        # layer norm
        self.ln_1 = LayerNorm(self.n_chans, bias=False)
        self.ln_2 = LayerNorm(self.n_chans, bias=False)

    def forward(self, x):
        B, T, C = x.shape # batch, seq_len, channels
        assert C == self.n_chans, "input channel dimension must match n_chans"
        #assert T % self.block_size == 0, "input length must be divisible by block size. T is {}".format(T)

        assert self.n_chans % self.n_heads == 0, "n_chans must be divisible by n_heads"

        # compute query, key, value for all heads in batch
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, C // n_heads)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, C // n_heads)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, C // n_heads)

        # scale query
        q = q * (C // self.n_heads) ** -0.5 # TODO check if this is correct

        # attention
        if self.causal and self.flash:
            y = torch.zeros_like(q)
            print("======================= Using Flash Attention: {}".format(self.flash))
            # flash attention is a bit faster but only works with dropout=0.0
            #attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)[0]
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
            if self.causal: # mask out future tokens
                attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            y = attn @ v # (B, n_heads, T, C // n_heads)

        # combine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # output projection
        y = self.c_proj(y)
        y = self.resid_drop(y)

        return y

class FeedForward(nn.Module):
    """ adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, n_chans) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.c_fc = nn.Linear(n_chans, 4 * n_chans, bias=False)
        self.c_proj = nn.Linear(4 * n_chans, n_chans, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.dropout(x) # TODO check if this should be after projection
        x = self.c_proj(x)
        return x

class DecoderBlock(nn.Module):
    """ adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, config):
        super().__init__()
        n_chans = data.X_DECODER_CHANNELS
        self.ln_1 = LayerNorm(n_chans, bias=False)
        self.attn = SelfAttention(config, causal=True)
        self.ln_2 = LayerNorm(n_chans, bias=False)
        # self.cross_attn = CrossAttention(n_chans, causal=False)
        self.mlp = FeedForward(n_chans)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        # x = x + self.cross_attn(self.ln_2(x), x)
        x = x + self.mlp(self.ln_2(x))
        return x


# === TRANSFORMER CLASS ===

class Transformer(nn.Module):
    """ adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            in_mlp = FeedForward(data.X_DECODER_CHANNELS),
            wpe = nn.Embedding(config.block_size, data.X_DECODER_CHANNELS), # positional embedding
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)]),
            ln_f = LayerNorm(data.X_DECODER_CHANNELS, bias=False),
        ))

        # initialize weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02, mean=0.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02, mean=0.0)
            if m.padding_idx is not None:
                nn.init.zeros_(m.weight[m.padding_idx])

    def forward(self, x_enc, x_dec):
        device = x_dec.device
        b, t = x_dec.shape[:2]
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # input MLP (for scaling bpm etc)
        x = self.transformer['in_mlp'](x_dec) 

        # add position embedding
        pos = torch.arange(t, device=device).unsqueeze(0).repeat(b, 1) # (b, t)
        pos_emb = self.transformer['wpe'](pos) # (b, t, n_chans)
        x = x + pos_emb
        x = self.transformer['drop'](x)
        
        # transformer blocks
        for block in self.transformer['h']:
            x = block(x)
        y_hat = self.transformer['ln_f'](x)

        return y_hat

    def loss(self, y_hat, y):
        hit_loss = F.mse_loss(y_hat[:, :, :12], y[:, :, :12]) # hits
        timing_loss = F.mse_loss(y_hat[:,:, 12] - y_hat[:,:, 13], y[:,:, 12] - y[:,:, 13]) # timing
        return hit_loss + timing_loss
        
    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        # decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, x_enc, x_dec, num_samples: int = 1):
        # generate predictions and append them to x_dec
        for _ in range(num_samples):
            x_dec = x_dec if x_dec.size(1) < self.block_size else x_dec[:, -self.block_size:]
            x_enc = x_enc if x_enc.size(1) < self.block_size else x_enc[:, -self.block_size:]
            y_hat = self(x_enc, x_dec)
            y_hat = y_hat[:, -1, :] # latest prediction = next step (b, n_chans)
            if x_dec.size(1) == 1 and x_dec[0, 0, 12] == -1.0: # first prediction
                x_dec = y_hat.unsqueeze(1) # (b, 1, n_chans)
            else:
                x_dec = torch.cat([x_dec, y_hat.unsqueeze(1)], dim=1) # (b, t+1, n_chans)
        return x_dec
  



# === TESTS ===
if __name__ == '__main__':
    test = torch.tensor([[  [0, 0, 0, 0, 0],
                            [42, 70, 120, 1, 0],
                            [36, 60, 120, 1, 0.5],
                            [38, 111, 140, 1.5, 0.33],
                            [42, 105, 140, 1.5, 0.33],
                            [36, 101, 140, 1.5, 0.66]]])
    #print(data.readScore(test).shape)
    #print(readScore(test)[:, :10, :])
    x_enc = data.readScore(test)
    x_dec = torch.randn(1, 1, 14)
    notes = data.readScoreLive(test[:,:3,:])
    #feat = x.squeeze(0)
    
    config = Config()
    m = Transformer(config)
    start = time.time()
    print("MODEL OUT:", m(x_enc, x_dec), m(x_enc, x_dec).shape)
    print("GENERATE:", m.generate(x_enc, x_dec, notes.shape[1]), m.generate(x_enc, x_dec, notes.shape[1]).shape)
    print(time.time() - start, "s")