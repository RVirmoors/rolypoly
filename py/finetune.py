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
import copy
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def getBatch(x_enc, y_hat, g, block_size: int):
    batch_size = y_hat.shape[0] // block_size
    if batch_size == 0:
        xe = x_enc.unsqueeze(0)
        xd = y_hat.unsqueeze(0)
        G  = g.unsqueeze(0)
    else:
        xe = torch.stack([x_enc[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        xd = torch.stack([y_hat[i*block_size:(i+1)*block_size] for i in range(batch_size)])
        G  = torch.stack([    g[i*block_size:(i+1)*block_size] for i in range(batch_size)])
    xe = data.dataScaleDown(xe)
    xd = data.dataScaleDown(xd)

    return xe, xd, G

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

def getLoss(D_hat, G_hat, G, V_hat, V, Follow: float):
    loss = torch.zeros(1, 1, constants.X_DECODER_CHANNELS)

    # remove loss for missing guitar notes
    mask = (G != 0)

    avg_D = torch.mean((D_hat * mask)[D_hat != 0])
    std_D = torch.std((D_hat * mask)[D_hat != 0])
    loss[:,:,1] = (avg_D - constants.gmd_tau_d_avg) ** 2 + (std_D - constants.gmd_tau_d_std) ** 2 # realised drum timing vs GMD
    loss[:,:,2] = F.mse_loss(V_hat * mask.unsqueeze(2), V * mask.unsqueeze(2)) # realised velocities vs score
    loss[:,:,3] = F.mse_loss(D_hat * mask, G) # realised drum timing vs guitar timing
    loss[:,:,4] = F.mse_loss(G_hat * mask, G) # realised vs predicted guitar timing

    loss[:,:,0] = (1 - Follow) * (loss[:,:,1] + 0.1 * loss[:,:,2]) + Follow * loss[:,:,3] + loss[:,:,4]   
    return loss

def finetune(m: model.Transformer, params: List[torch.Tensor], x_enc, x_dec, y_hat, Follow:float = 0.5):
    x_enc = x_enc.squeeze() # [seq_len, 12]
    g = x_dec[:,:,constants.INX_TAU_G].squeeze() # [seq_len]
    xd = torch.zeros(1, 1, constants.X_DECODER_CHANNELS)
    # d, v = generateGMD(m, x_enc, xd) # [seq_len]
    y_hat = y_hat.squeeze() # [seq_len, 14]
    block_size = m.block_size

    optimizer = adam_ts.Adam(params, lr=constants.lr, betas=(constants.beta1, constants.beta2), weight_decay=constants.weight_decay)
    X_enc, X_dec, G = getBatch(x_enc, y_hat, g, block_size) # add batch dimension
    D_hat = torch.zeros_like(G)
    G_hat = torch.zeros_like(G)

    loss = torch.zeros(1, 0, constants.X_DECODER_CHANNELS)
    best_loss = 1000
    best_params = params.copy()
    losses = torch.zeros(1, 1, constants.X_DECODER_CHANNELS)
    for epoch in range(constants.epochs):
        Y_hat = m.forward(X_enc, X_dec)
        D_hat, G_hat = getTimings(X_dec, Y_hat) # we already have G from getBatch
        V = X_enc[:,:,:9]
        V_hat = Y_hat[:,:,:9]

        loss = getLoss(D_hat, G_hat, G, V_hat, V, Follow)
        losses = torch.cat((losses, loss), dim=1)
        X_dec = Y_hat.detach().clone() # for the next step

        if (loss[:,:,0].item() < best_loss):
            best_loss = loss[:,:,0].item()
            best_params = params.copy()
            # print("best loss", best_loss)

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

    # update model parameters with best params
    for i in range(len(params)):
        params[i] = best_params[i]

    # diagnostics: D_hat, G_hat, G
    diag = torch.stack((D_hat, G_hat, G), dim=2)
    diag = torch.cat((diag, torch.zeros(diag.shape[0], diag.shape[1], 11)), dim=2)
    print("              D_hat          G_hat         G\n", diag[0, :10, :3])

    return m, params, losses, diag



# ================ TEST ============

def run_gmd(x_take):
    xd, xe, _ = train_gmd.getTrainDataFromX_take(x_take)
    x_dec = xd[0].unsqueeze(0).unsqueeze(0)
    x_enc = xe.unsqueeze(0)
    y_hat = x_dec.clone().detach()
    print("first x_dec:\n", x_dec[0, :3], x_dec.shape)
    print("first x_enc:\n", x_enc[0, :3], x_enc.shape)
    # generate
    for i in range(13):
    # for i in range(x_enc.shape[1] - 1):
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
        # if torch.rand(1) < 0.4:
        #     guit = torch.tensor([[[666, torch.rand(1)*100, 0,0,0]]])
        #     guit[:,:,2:5] = x_dec[:, -1, 9:12]
        #     data.readLiveOnset(guit, x_dec, x_enc)
    
    return x_enc, x_dec, y_hat


if __name__ == '__main__':
    # load pretrained model
    checkpoint = torch.load('out/ckpt.pt', map_location=device)
    config = checkpoint['config']
    m = model.Transformer(config)
    m.load_state_dict(torch.load('out/model_best.pt', map_location=device))
    print("Loaded pretrained model:", checkpoint['iter_num'], "epochs, loss:", checkpoint['best_val_loss'].item())

    # checkpoint = torch.load('out/ckpt.pt', map_location=device)
    # config = checkpoint['config']
    # m = model.Transformer(config)
    # m.load_state_dict(torch.jit.load('../help/model.pt', map_location=device).pretrained.state_dict())
    # print("Loaded pretrained model:", type(m))
    m.eval()

    # simulate live run
    x_take = data.loadX_takeFromCSV('gmd.csv')
    m.eval()
    x_enc, x_dec, y_hat = run_gmd(x_take)
    print("x_dec after run:\n", x_dec, x_dec.shape)
    # print("y_hat after run:\n", y_hat, y_hat.shape)

    # finetune
    for _ in range(5):
        x_dec[0, 3, constants.INX_TAU_G] = -0.02
        x_dec[0, 5, constants.INX_TAU_G] = -0.02
        x_dec[0, 7, constants.INX_TAU_G] = -0.02
        x_dec[0, 9, constants.INX_TAU_G] = -0.02
        x_dec[0, 11, constants.INX_TAU_G] = -0.02
        x_dec[0, 13, constants.INX_TAU_G] = -0.02
        x_dec[0, 4, constants.INX_TAU_G] = 0.01
        x_dec[0, 6, constants.INX_TAU_G] = 0.01
        x_dec[0, 8, constants.INX_TAU_G] = 0.01
        x_dec[0, 10, constants.INX_TAU_G] = 0.01
        x_dec[0, 12, constants.INX_TAU_G] = 0.01
        m.train()
        t0 = time.time()
        torch.set_printoptions(sci_mode=False, linewidth=200, precision=6)
        m, _, _ ,_ = finetune(m, list(m.parameters()), x_enc[:,:14], x_dec, y_hat, Follow=0.999)    
        # finetune(m, list(m.parameters()), x_enc, x_dec, y_hat, Follow=0.4)
        t1 = time.time()
        print("finetune took", t1-t0, "s")

        torch.set_printoptions(sci_mode=False, linewidth=200, precision=2)
        m.eval()
        x_enc, x_dec, y_hat = run_gmd(x_take)

    print("x_dec after run:\n", x_dec, x_dec.shape)
    # print("y_hat after run:\n", y_hat, y_hat.shape)
