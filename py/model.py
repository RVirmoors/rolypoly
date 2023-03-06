"""
Rolypoly timing model
2023 rvirmoors

Very much inspired by A Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.set_printoptions(sci_mode=False, linewidth=200, precision=2)

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
    arch = 'ed' # 'd' for decoder-only, 'ed' for encoder-decoder
    n_layers = 6 # 10 # number of block layers
    d_model = 64 # 128 # number of channels in the model
    block_size = 16 # number of hits in a block
    dropout = 0.15 # dropout rate
    max_seq_len = 1000 # max sequence length

# === HELPER CLASSES FOR TRANSFORMER ===

class PositionalEncodingSimple(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.embed = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, x, t = 0):
        #print("embedding from", t, "to", self.max_seq_len)
        pos = torch.arange(t, self.max_seq_len, dtype=torch.long).to(x.device).unsqueeze(0)
        #print("pos", pos.shape, pos[:, :4])
        return self.embed(pos)

class PositionalEncoding(nn.Module):
    """ Positional Encoding """
    def __init__(self):
        super().__init__()

    def forward(self, x, position):
        # add positional encoding
        b, seq_len, dim_model = x.shape
        # print("DIM", x.shape, position.shape)
        pe = torch.zeros_like(x).to(x.device)
        position = position[:, :seq_len]
        position = position.view(b, seq_len, 1) # (b, seq_len, 1)
        #print("POS", position, position.shape)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(128.0) / dim_model)).to(x.device).view(1, 1, -1) # (b, 1, dim_model/2)
        pe[:, :, 0::2] = torch.sin(position * math.pi * 2 * div_term) # (1, seq_len, dim_model/2)
        pe[:, :, 1::2] = torch.cos(position * math.pi * 2 * div_term)
        # print("pe", pe.shape)
        return pe

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
    # TODO: replace with nn.MultiheadAttention ?

    def __init__(self, config, causal=True) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.block_size = config.block_size
        self.causal = causal
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))

        self.n_chans = config.d_model       # 128
        self.n_heads = config.d_model // 4 # 4

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_chans, 3 * self.n_chans, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_chans, self.n_chans, bias=False)
        # regularization
        self.attn_drop = nn.Dropout(self.dropout)
        self.resid_drop = nn.Dropout(self.dropout)

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

class CrossAttention(nn.Module):
    """ inputs: dec prev. layer & enc output, (batch, seq_len, channels) each
        output: y projection (batch, seq_len, channels)

    http://nlp.seas.harvard.edu/annotated-transformer/
    In “encoder-decoder attention” layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence.

    - adapted from SelfAttention above"""

    def __init__(self, config) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.block_size = config.block_size
        self.n_chans = config.d_model       # 128
        self.n_heads = config.d_model // 4 # 4

        # query, key, value projections for all heads, but in a batch
        # query is from previous decoder layer, key and value are from encoder output
        self.attnQ = nn.Linear(self.n_chans, self.n_chans, bias=False)
        self.attnKV = nn.Linear(self.n_chans, 2 * self.n_chans, bias=False)
        # output projection
        self.proj = nn.Linear(self.n_chans, self.n_chans, bias=False)
        # regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, enc_out):
        B, T, C = x.shape # batch, seq_len, channels
        Be, Te, Ce = enc_out.shape # batch, seq_len, channels
        assert Be == B, "batch size of encoder output must match decoder input"
        assert Ce == C, "channel size of encoder output must match decoder input"

        # compute query, key, value for all heads in batch
        q = self.attnQ(x) # (B, T, C)
        k, v = self.attnKV(enc_out).chunk(2, dim=-1) # (B, T, C) each
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, C // n_heads)
        k = k.view(B, Te, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, C // n_heads)
        v = v.view(B, Te, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n_heads, T, C // n_heads)

        # scale query
        q = q * (C // self.n_heads) ** -0.5 # TODO check if this is correct

        # attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = attn @ v # (B, n_heads, T, C // n_heads)

        # combine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # output projection
        y = self.proj(y)
        y = self.resid_drop(y)
        return y

class FeedForward(nn.Module):
    """ adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, n_chans) -> None:
        super().__init__()
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
        n_chans = config.d_model # 128
        self.arch = config.arch
        self.ln_1 = LayerNorm(n_chans, bias=False)
        self.attn = SelfAttention(config, causal=True)
        self.ln_2 = LayerNorm(n_chans, bias=False)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = LayerNorm(n_chans, bias=False)
        self.mlp = FeedForward(n_chans)

    def forward(self, x, enc_out):
        x = x + self.attn(self.ln_1(x))
        if self.arch == 'ed': # encoder-decoder attention
            x = x + self.cross_attn(self.ln_2(x), enc_out)
        x = x + self.mlp(self.ln_3(x))
        return x

class EncoderBlock(nn.Module):
    """ adapted from DecoderBlock above"""
    
    def __init__(self, config):
        super().__init__()
        n_chans = config.d_model # 128
        self.ln_1 = LayerNorm(n_chans, bias=False)
        self.attn = SelfAttention(config, causal=False)
        self.ln_2 = LayerNorm(n_chans, bias=False)
        self.mlp = FeedForward(n_chans)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# === TRANSFORMER CLASS ===

class Transformer(nn.Module):
    """adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py"""

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.arch = config.arch
        in_out_chans = data.X_DECODER_CHANNELS # 14
        enc_in_chans = data.X_ENCODER_CHANNELS # 12

        if self.arch == 'd':
            self.transformer = nn.ModuleDict(dict(
                in_dec = nn.Linear(in_out_chans, config.d_model),
                wpe_dec = PositionalEncoding(),
                drop_dec = nn.Dropout(0.1),
                h_dec = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)]),
                ln_f = LayerNorm(config.d_model, bias=False),
                head = nn.Linear(config.d_model, in_out_chans),
            ))
        elif self.arch == 'ed':
            self.transformer = nn.ModuleDict(dict(
                in_enc = nn.Linear(enc_in_chans, config.d_model),               
                wpe_enc = PositionalEncodingSimple(config),
                drop_enc = nn.Dropout(config.dropout),
                h_enc = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)]),
                proj_enc = FeedForward(config.d_model),

                in_dec = nn.Linear(in_out_chans, config.d_model),
                wpe_dec = PositionalEncodingSimple(config),
                drop_dec = nn.Dropout(config.dropout),
                h_dec = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)]),

                ln_f = LayerNorm(config.d_model, bias=False),
                head = nn.Linear(config.d_model, in_out_chans),
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

    def forward(self, x_enc, x_dec, t = 0):
        # predicting with x_dec starting at timestep t in the song
        device = x_dec.device
        b, seq_len = x_dec.shape[:2]
        assert seq_len <= self.block_size, f"Cannot forward sequence of length {seq_len}, block size is only {self.block_size}"
        
        if x_enc.shape[1] == 0: # error handling for empty encoder input
            x_enc = torch.zeros((x_enc.shape[0], 1, x_enc.shape[2]), device=device)
        
        x_enc = x_enc.clone().detach()
        x_dec = x_dec.clone().detach()
        bar_pos = x_enc[:, :, data.INX_BAR_POS] # get bar position from encoder input
        bar_num = bar_pos // 1 # get bar numbers
        # print ("bar_num", bar_num, bar_num.shape)
        #print ("bar_pos", bar_pos, bar_pos.shape)
        bar_pos_dec = bar_pos.detach().clone()[:, t:]
        #print("bar_pos_dec", bar_pos_dec, bar_pos_dec.shape)

        bp = bar_pos.detach().clone()
        x_enc[:, :, data.INX_BAR_POS] = torch.frac(bp) # set bar position to fraction of bar
        x_dec[:, :, data.INX_BAR_POS] = torch.frac(bp[:,:seq_len]) # set bar position to fraction of bar
        x_enc[:, :, data.INX_BAR_POS] = torch.cos(x_enc[:, :, data.INX_BAR_POS] * math.pi) # set bar position to cos of fraction of bar
        x_dec[:, :, data.INX_BAR_POS] = torch.cos(x_dec[:, :, data.INX_BAR_POS] * math.pi) # set bar position to cos of fraction of bar

        # add position embedding (ENCODER)
        if self.arch == 'ed':
            x_enc = self.transformer.in_enc(x_enc)
            #pos_emb_enc = self.transformer.wpe_enc(x_enc, bar_pos) 
            pos_emb_enc = self.transformer.wpe_enc(x_enc)
            x_enc = x_enc + pos_emb_enc[:, :x_enc.shape[1]]
            x_enc = self.transformer.drop_enc(x_enc)

        # add position embedding (DECODER)
        x_dec = self.transformer.in_dec(x_dec)
        # pos_emb_dec = self.transformer.wpe_dec(x_dec, bar_pos_dec)
        pos_emb_dec = self.transformer.wpe_dec(x_dec, t)
        x_dec = x_dec + pos_emb_dec[:, :x_dec.shape[1]]
        x_dec = self.transformer.drop_dec(x_dec)
        
        # transformer blocks (ENCODER)
        if self.arch == 'ed':
            for block in self.transformer.h_enc:
                x_enc = block(x_enc)
            enc_out = self.transformer.proj_enc(x_enc) # (b, t, n_decoder_chans)
        else:
            enc_out = self.transformer.proj_enc(torch.zeros_like(x_dec, device=device))

        # enc_out = enc_out + pos_emb_dec # TODO: check if this might help
        
        # transformer blocks (DECODER)
        for block in self.transformer.h_dec:
            x_dec = block(x_dec, enc_out)
        y_hat = self.transformer.ln_f(x_dec)

        y_hat = self.transformer.head(y_hat)
        # y_hat[:, -1, data.INX_BAR_POS] = torch.acos(y_hat[:, -1, data.INX_BAR_POS]) / math.pi # convert bar position back to fraction of bar
        # y_hat[:, -1, data.INX_BAR_POS] = y_hat[:, -1, data.INX_BAR_POS] + bar_num[:,seq_len-1] # add current bar back to output
        y_hat[:,:,:9] = torch.sigmoid(y_hat[:,:,:9]) # apply sigmoid to hits
        y_hat[:, :, 9:12] = torch.tanh(y_hat[:, :, 9:12]) # apply tanh to position
        y_hat[:, :, 12:] = torch.tanh(y_hat[:, :, 12:]) # apply tanh to timing
        return y_hat

    def loss(self, y_hat, y):
        # for hits where y is 0, weight the loss by 0.2
        # for hits where y is 1, weight the loss by 1
        
        _y = y.clone().detach()
        mask = torch.where(_y[:, :, :9] == 0, 0.2, 1.0) # create mask
        hit_loss = F.mse_loss(y_hat[:, :, :9], y[:, :, :9], reduction='none') # calculate binary cross entropy loss
        weighted_hit_loss = hit_loss * mask # apply weighting
        hit_loss = torch.mean(weighted_hit_loss) # calculate mean of weighted loss

        #hit_loss = F.mse_loss(y_hat[:, :, :9], y[:, :, :9]) # hits
        pos_loss = F.mse_loss(y_hat[:, :, 9:12], y[:, :, 9:12]) # position
        timing_loss = F.mse_loss(y_hat[:,:, 12] - y_hat[:,:, 13], y[:,:, 12] - y[:,:, 13]) # timing
        #print("LOSS\ny_hat\n", y_hat[-1,-2], y_hat.shape, "\ny\n", y[-1,-2], y.shape, "\nhit_loss", hit_loss, "timing_loss", 100 * timing_loss, "pos_loss", 0.1 * pos_loss)
        return hit_loss + 0.1 * pos_loss + 100 * timing_loss # weigh timing loss higher
        
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
            # crop inputs to block size
            t = x_dec.size(1) - 1 # current time step
            xd = x_dec if x_dec.size(1) < self.block_size else x_dec[:, -self.block_size:]
            print("==current time step:", t, "==")

            _xe = x_enc.clone().detach()
            _xe = data.dataScaleUp(_xe)
            print("x_enc:\n", _xe[0, :t+8, 11], "...", _xe.shape)

            _xd = xd.clone().detach()
            _xd = data.dataScaleUp(_xd)
            print("x_dec:\n", _xd[0, :, 11], _xd.shape)

            # generate prediction
            dec_start = 0 if t < self.block_size else t - self.block_size + 1
            # print("dec_start", dec_start, "t", t, "block_size", self.block_size)
            y_hat = self(x_enc, xd, dec_start) # (b, t, n_chans)
            y_hat = y_hat[:, -1, :] # latest prediction = next step (b, n_chans)

            # # append prediction to x_dec
            # print(x_dec.shape, y_hat.unsqueeze(1).shape)
            x_dec = torch.cat([x_dec, y_hat.unsqueeze(1)], dim=1) # (b, t+1, n_chans)

        return x_dec
  

# === TESTS ===
if __name__ == '__main__':
    test = torch.tensor([[  [0, 0, 0, 0, 0],
                            [42, 70, 120, 1, 0],
                            [36, 60, 120, 1, 0.5],
                            [38, 111, 140, 1.5, 1.33],
                            [42, 105, 140, 1.5, 1.33],
                            [36, 101, 140, 1.5, 1.66]]])
    x_enc = data.readScore(test)
    print("X_ENC:", x_enc, x_enc.shape)
    x_dec = torch.randn(1, 1, 14)
    notes = data.readScoreLive(test[:,:3,:])
    
    config = Config()
    m = Transformer(config)
    start = time.time()
    x_enc = data.dataScaleDown(x_enc)
    x_dec = data.dataScaleDown(x_dec)
    print("GENERATE:\n", m.generate(x_enc, x_dec, notes.shape[1]))
    print(time.time() - start, "s")

    # print(m.state_dict())