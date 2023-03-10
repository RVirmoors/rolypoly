"""Rolypoly Python implementation
2023 rvirmoors

Finetune the pretrained network using the received X_dec and X_enc data.
Called from the Max object via export.py
"""

import torch
from torch.nn import functional as F
from typing import List, Tuple

import data
import constants
import train_gmd
import model
import adam_ts

import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def getBatch(x_enc, y_hat, g, d, v, block_size: int):
    batch_size = y_hat.shape[0] // block_size
    if batch_size == 0:
        xe = x_enc.unsqueeze(0)
        xd = y_hat.unsqueeze(0)
        G  = g.unsqueeze(0)
        D  = d.unsqueeze(0)
        V  = v.unsqueeze(0)
    else:
        xe = torch.stack([x_enc[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        xd = torch.stack([y_hat[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        G  = torch.stack([    g[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        D  = torch.stack([    d[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        V  = torch.stack([    v[i*block_size:(i+1)*block_size] for i in range(batch_size)])
    xe = data.dataScaleDown(xe)
    xd = data.dataScaleDown(xd)

    return xe, xd, G, D, V

def getTimings(x_dec, y_hat):
    D_hat = x_dec[:,:,constants.INX_TAU_D]
    G_hat = y_hat[:,:,constants.INX_TAU_G]
    return D_hat, G_hat

def generateGMD(model: model.Transformer, x_enc, xd):
    # generate drum timings for a guitarless take
    xe = x_enc.detach().clone().unsqueeze(0) # [1, seq_len, 12]
    x_dec = model.generate(xe, xd, num_samples=xe.shape[1]-1)
    d = x_dec[0,:,constants.INX_TAU_D]
    v = x_dec[0,:,:9]
    return d, v

def getLoss(D_hat, D, G_hat, G, V_hat, V, Follow: float):
    loss = torch.zeros(1, 1, constants.X_DECODER_CHANNELS)
    loss[:,:,1] = F.mse_loss(D_hat, D) # realised drum timing vs GMD
    loss[:,:,2] = F.mse_loss(V_hat, V) # realised velocities vs GMD
    loss[:,:,3] = F.mse_loss(D_hat, G - G_hat) # realised drum timing vs guitar timing
    loss[:,:,4] = F.mse_loss(G_hat, G) # realised vs predicted guitar timing
    loss[:,:,0] = (1 - Follow) * (loss[:,:,1] + 0.1 * loss[:,:,2]) + Follow * loss[:,:,3] + 1.0 * loss[:,:,4] 
    return loss

def finetune(m: model.Transformer, params: List[torch.Tensor], x_enc, x_dec, y_hat, Follow:float = 0.5):
    x_enc = x_enc.squeeze() # [seq_len, 12]
    g = x_dec[:,:,constants.INX_TAU_G].squeeze() # [seq_len]
    xd = torch.zeros(1, 1, constants.X_DECODER_CHANNELS)
    d, v = generateGMD(m, x_enc, xd) # [seq_len]
    y_hat = y_hat.squeeze() # [seq_len, 14]
    block_size = m.block_size

    # m.to(device)
    # m.train()
    optimizer = adam_ts.Adam(params, lr=constants.lr, betas=(constants.beta1, constants.beta2))
    # optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = torch.jit.script(optimizer)
    X_enc, X_dec, G, D, V = getBatch(x_enc, y_hat, g, d, v, block_size) # add batch dimension

    loss = torch.zeros(1, 1, constants.X_DECODER_CHANNELS)
    losses = torch.zeros(1, 1, constants.X_DECODER_CHANNELS)
    epochs = 12 + x_enc.shape[0] // 30 # 12 + 1 epoch per 30 steps in the input sequence
    for epoch in range(epochs):
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        Y_hat = m.forward(X_enc, X_dec)
        D_hat, G_hat = getTimings(X_dec, Y_hat) # we already have G & D from getBatch
        V_hat = Y_hat[:,:,:9]
        # print("D_hat", D_hat[0,:10], "D", D[0,:10], "G_hat", G_hat[0,:10], "G", G[0,:10])

        loss = getLoss(D_hat, D, G_hat, G, V_hat, V, Follow)
        losses = torch.cat((losses, loss), dim=1)
        X_dec = Y_hat.detach().clone() # for the next step

        optimizer.zero_grad()
        loss[:,:,0].backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0) # TODO: make this work in torchscript

        # create list of gradients
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad)
            else:
                grads.append(torch.zeros_like(p))

        optimizer.step(params, grads)
        print("epoch", epoch, "loss", loss[:,:,:5])

    return m, losses



# ================ TEST ============

def run_gmd(x_take):
    xd, xe, _ = train_gmd.getTrainDataFromX_take(x_take)
    x_dec = xd[0].unsqueeze(0).unsqueeze(0)
    x_enc = xe.unsqueeze(0)
    y_hat = x_dec.clone().detach()
    print("first x_dec:\n", x_dec[0, :3], x_dec.shape)
    print("first x_enc:\n", x_enc[0, :3], x_enc.shape)
    # generate
    # for i in range(35):
    for i in range(x_enc.shape[1] - 1):
        xd = x_dec.clone().detach()
        xe = x_enc.clone().detach()
        data.dataScaleDown(xd)
        data.dataScaleDown(xe)
        x_dec = m.generate(xe, xd, 1)
        data.dataScaleUp(x_dec)
        y_hat = torch.cat((y_hat, x_dec[:, -1:]), dim=1)
        x_dec[:, -1:, 13] = 0
        x_dec[:, -1, 9:12] = x_enc[:, i+1, 9:12]
        # set non x_enc notes to zero
        x_dec[:,:,:x_enc.shape[2]][x_enc[:,:x_dec.shape[1]] == 0] = 0
        if torch.rand(1) < 0.4:
            guit = torch.tensor([[[666, torch.rand(1)*100, 0,0,0]]])
            guit[:,:,2:5] = x_dec[:, -1, 9:12]
            data.readLiveOnset(guit, x_dec, x_enc)
    
    return x_enc, x_dec, y_hat


if __name__ == '__main__':
    # load pretrained model
    checkpoint = torch.load('out/ckpt.pt', map_location=device)
    config = checkpoint['config']
    m = model.Transformer(config)
    m.load_state_dict(torch.load('out/model_best.pt', map_location=device))
    print("Loaded pretrained model:", checkpoint['iter_num'], "epochs, loss:", checkpoint['best_val_loss'].item())

    # simulate live run
    x_take = data.loadX_takeFromCSV('gmd.csv')
    x_enc, x_dec, y_hat = run_gmd(x_take)
    print("x_dec after run:\n", x_dec, x_dec.shape)
    print("y_hat after run:\n", y_hat, y_hat.shape)

    # finetune
    t0 = time.time()
    torch.set_printoptions(sci_mode=False, linewidth=200, precision=6)
    # finetune(m, list(m.parameters()), x_enc[:,:36], x_dec, y_hat, Follow=0.5)    
    finetune(m, list(m.parameters()), x_enc, x_dec, y_hat, Follow=0.4)
    t1 = time.time()
    print("finetune took", t1-t0, "s")

    torch.set_printoptions(sci_mode=False, linewidth=200, precision=2)
    x_enc, x_dec, y_hat = run_gmd(x_take)
    print("x_dec after run:\n", x_dec, x_dec.shape)
    print("y_hat after run:\n", y_hat, y_hat.shape)
