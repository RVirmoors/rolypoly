"""Rolypoly Python implementation
2023 rvirmoors

Train network on a csv file of a performance.

Code heavily inspired by https://github.com/karpathy/nanoGPT/blob/master/train.py
"""


import torch, torch.nn as nn
import numpy as np
import math
import time
import os

import data # data helper methods

# I/O
out_dir = 'out'
eval_interval = 25
log_interval = 10
eval_iters = 10 # 200
eval_only = False # if True, script exits right after the first eval
init_from = 'scratch' # 'scratch' or 'resume'

os.makedirs(out_dir, exist_ok=True)

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = False # use PyTorch 2.0 to compile the model to be faster
gradient_accumulation_steps = 5 # how many steps to accumulate gradients over before performing a step

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def getBatch(split, train_xd, val_xd, batch_size, block_size, train_xe=None, val_xe=None):
    xd = train_xd if split == 'train' else val_xd
    if train_xe is None:
        xe = xd
    else:
        xe = train_xe if split == 'train' else val_xe

    take_i = np.random.randint(0, len(xd), (batch_size))
    ix = [np.random.randint(0, xd[i].shape[0] - block_size) for i in take_i]
    x_dec = torch.stack([xd[take_i[i]][ix[i]:ix[i]+block_size] for i in take_i])
    x_enc = torch.stack([xe[take_i[i]][ix[i]:ix[i]+block_size] for i in take_i])
    y = torch.stack([xd[take_i[i]][ix[i]+1:ix[i]+block_size+1] for i in take_i])

    if 'cuda' in device:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x_enc, x_dec, y = x_enc.pin_memory().to(device, non_blocking=True), x_dec.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
    return x_enc, x_dec, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, train_xd, val_xd, batch_size, block_size, train_xe=None, val_xe=None):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X_enc, X_dec, Y = getBatch(split, train_xd, val_xd, batch_size, block_size, train_xe, val_xe)
            X_enc, X_dec, Y = data.dataScaleDown(X_enc), data.dataScaleDown(X_dec), data.dataScaleDown(Y)
            y_hat = model(X_enc, X_dec)
            loss = model.loss(y_hat, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, config, load_model, epochs, train_xd, val_xd, batch_size, train_xe=None, val_xe=None):
    block_size = config.block_size

    if load_model:
        print("Resuming training from", load_model, "...")
        checkpoint = torch.load(load_model, map_location=device)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
    if load_model:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if compile:
        print("compiling the model... (takes ~a minute)")
        unoptimized_model = m
        m = torch.compile(m) # requires PyTorch 2.0

    iter_num = 0
    best_val_loss = 1e10
    
    scaler = torch.cuda.amp.GradScaler(enabled = (dtype == 'float16'))

    # training loop
    X, Y = getBatch('train', train_xd, val_xd, batch_size, block_size, train_xe, val_xe) # get first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    #for epoch in range(epochs):
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model, train_xd, val_xd, batch_size, block_size, train_xe, val_xe)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), f'{out_dir}/model_best.pt')
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            Y_hat = model(X, Y)
            loss = model.loss(Y_hat, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = getBatch('train', train_xd, val_xd, batch_size, block_size, train_xe, val_xe)
            if dtype == 'float16':
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # update the parameters
        if dtype == 'float16':
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        # flush the gradients
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            lossf = loss.item()
            print(f"step {iter_num}: train loss {lossf:.4f}, lr {lr:.4f}, {dt:.2f}s")

        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    return model