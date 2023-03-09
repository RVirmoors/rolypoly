"""Rolypoly Python implementation
2023 rvirmoors

Finetune the pretrained network using the received X_dec and X_enc data.
Called from the Max object via export.py
"""

import torch
from torch.nn import functional as F

import data
import constants
import train
import train_gmd
import model

import time
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
lr = 6e-4 # learning rate

def getBatch(x_enc, y_hat, g, d, block_size):
    batch_size = math.floor(y_hat.shape[0] / block_size)
    if batch_size == 0:
        xe = x_enc.unsqueeze(0)
        xd = y_hat.unsqueeze(0)
        G  = g.unsqueeze(0)
        D  = d.unsqueeze(0)
    else:
        xe = torch.stack([x_enc[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        xd = torch.stack([y_hat[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        G  = torch.stack([    g[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        D  = torch.stack([    d[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        xe = torch.cat((xe, x_enc[batch_size*block_size:].unsqueeze()), dim=0) # [batch_size, block_size, 12]
        xd = torch.cat((xd, y_hat[batch_size*block_size:].unsqueeze()), dim=0) # [batch_size, block_size, 14]
        G  = torch.cat((G,      g[batch_size*block_size:].unsqueeze()), dim=0) # [batch_size, block_size]
        D  = torch.cat((D,      d[batch_size*block_size:].unsqueeze()), dim=0) # [batch_size, block_size]
    xe = data.dataScaleDown(xe)
    xd = data.dataScaleDown(xd)

    if 'cuda' in device:
        # pin arrays, which allows us to move them to GPU asynchronously (non_blocking=True)
        xe, xd, G, D = xe.pin_memory().to(device, non_blocking=True), xd.pin_memory().to(device, non_blocking=True), G.pin_memory().to(device, non_blocking=True), D.pin_memory().to(device, non_blocking=True)
    else:
        xe, xd, G, D = xe.to(device), xd.to(device), G.to(device), D.to(device)
    return xe, xd, G, D

def getTimings(x_dec, y_hat):
    D_hat = x_dec[:,:,constants.INX_TAU_D]
    G_hat = y_hat[:,:,constants.INX_TAU_G]
    return D_hat, G_hat

def generateGMD(model, x_enc, xd):
    # generate drum timings for a guitarless take
    xe = x_enc.detach().clone().unsqueeze(0) # [1, seq_len, 12]
    x_dec = model.generate(xe, xd, num_samples=xe.shape[1])
    d = x_dec[0,1:,constants.INX_TAU_D]
    return d

def getLoss(D_hat, D, G_hat, G, Follow):
    loss = torch.zeros(1, 1, constants.X_DECODER_CHANNELS)
    loss[:,:,1] = F.mse_loss(D_hat, D)
    loss[:,:,2] = F.mse_loss(D_hat, G - G_hat)
    loss[:,:,3] = F.mse_loss(G, G_hat)
    loss[:,:,0] = (1 - Follow) * loss[:,:,1] + Follow * loss[:,:,2] + 1.0 * loss[:,:,3]
    return loss

def finetune(model, x_enc, x_dec, y_hat, Follow:float = 0.5):
    x_enc = x_enc.squeeze() # [seq_len, 12]
    g = x_dec[:,:,constants.INX_TAU_G].squeeze() # [seq_len]
    xd = torch.zeros(1, 1, constants.X_DECODER_CHANNELS)
    d = generateGMD(model, x_enc, xd) # [seq_len]
    y_hat = y_hat.squeeze() # [seq_len, 14]
    block_size = model.block_size

    model.to(device)
    optimizer = model.configure_optimizers(weight_decay, lr, (beta1, beta2), device)

    X_enc, X_dec, G, D = getBatch(x_enc, y_hat, g, d, block_size)

    t0 = time.time()
    for epoch in range(5):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        Y_hat = model(X_enc, X_dec)
        D_hat, G_hat = getTimings(X_dec, Y_hat) # we already have G & D from getBatch

        loss = getLoss(D_hat, D, G_hat, G, Follow)
        X_dec = Y_hat.detach().clone() # for the next step
        loss[:,:,0].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        t1 = time.time()
        torch.set_printoptions(sci_mode=False, linewidth=200, precision=4)
        print("epoch", epoch, "loss", loss[:,:,:4], "time", t1-t0)


    return model, loss



# ================ TEST ============

if __name__ == '__main__':
    # load pretrained model
    checkpoint = torch.load('out/ckpt.pt', map_location=device)
    config = checkpoint['config']
    m = model.Transformer(config)
    m.load_state_dict(torch.load('out/model_best.pt', map_location=device))
    print("Loaded pretrained model:", checkpoint['iter_num'], "epochs, loss:", checkpoint['best_val_loss'].item())

    # simulate live run
    x_take = data.loadX_takeFromCSV('gmd.csv')
    xd, xe, _ = train_gmd.getTrainDataFromX_take(x_take)
    x_dec = xd[0].unsqueeze(0).unsqueeze(0)
    x_enc = xe.unsqueeze(0)
    y_hat = x_dec.clone().detach()
    print("first x_dec:\n", x_dec[0, :3], x_dec.shape)
    print("first x_enc:\n", x_enc[0, :3], x_enc.shape)
    # generate
    for i in range(3):#x_enc.shape[1]):
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
    xd = x_dec.clone().detach()
    print("x_dec after run:\n", xd, xd.shape)
    print("y_hat after run:\n", y_hat, y_hat.shape)

    # finetune
    finetune(m, x_enc[0,:4], x_dec, y_hat, Follow=0.5)