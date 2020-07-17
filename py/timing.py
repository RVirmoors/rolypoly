"""
Rolypoly timing model
2020 rvirmoors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import datetime
import time
import os
import copy

from constants import ROLAND_DRUM_PITCH_CLASSES

# Helper libraries
import random
# from tqdm import tqdm

# Deterministic results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEBUG = False

feat_vec_size = len(ROLAND_DRUM_PITCH_CLASSES) + 4 + 1

loss = 0

# METHODS FOR BUILDING X, Y
# =========================


def addRow(featVec, y_hat, d_g_diff, X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths):
    # if new bar, finish existing sequence and start a new one
    if featVec[12] <= X[s_i][h_i][12] and h_i:
        if s_i >= 0:  # s_i is init'd as -1, so first note doesn't trigger:
            # move delay to first hit in new seq
            Y_hat[s_i + 1][0] = Y_hat[s_i][h_i + 1]
            # last hit plus one doesn't make sense
            Y_hat[s_i][h_i + 1] = 0
            if DEBUG:
                print("added bar #", s_i, "w/", int(X_lengths[s_i]), "hits.")
                print("Y_hat for seq:", Y_hat[s_i][:int(X_lengths[s_i])])
                print("==========")
        s_i += 1
        h_i = 0
        X[s_i][0] = featVec          # first hit in new seq
        Y_hat[s_i][1] = y_hat        # delay for next hit
        diff_hat[s_i][0] = d_g_diff  # drum-guitar diff for this hit
    else:
        h_i += 1
        X[s_i][h_i] = featVec           # this hit
        Y_hat[s_i][h_i + 1] = y_hat     # delay for next hit
        diff_hat[s_i][h_i] = d_g_diff   # drum-guitar diff for this hit
        X_lengths[s_i] = h_i + 1
    return X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths


def prepare_X(X, X_lengths, Y_hat, diff_hat, batch_size):
    """
    Pad short sequences
    https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    """
    longest_seq = int(max(X_lengths))
    if DEBUG:
        print("longest: ", longest_seq, " | batch size: ", batch_size)
    padded_X = np.zeros((batch_size, longest_seq, feat_vec_size))
    padded_Y_hat = np.zeros((batch_size, longest_seq))
    padded_diff_hat = np.zeros_like(padded_Y_hat)
    for i, x_len in enumerate(X_lengths[:batch_size]):
        x_len = int(x_len)
        # print("i", i, "seq length", x_len)
        sequence = X[i]
        padded_X[i, 0:x_len] = sequence[:x_len]
        sequence = Y_hat[i]
        padded_Y_hat[i, 0:x_len] = sequence[:x_len]
        sequence = diff_hat[i]
        padded_diff_hat[i, 0:x_len] = sequence[:x_len]
    X = padded_X
    Y_hat = padded_Y_hat
    diff_hat = padded_diff_hat
    X_lengths = X_lengths[:batch_size]
    X = torch.Tensor(X)
    X_lengths = torch.LongTensor(X_lengths)
    return X, X_lengths, Y_hat, diff_hat


def prepare_Y(X_lengths, diff_hat, Y_hat, Y, style='constant', value=None):
    """
    Computes Y, the "correct" values to be used in the MSE loss function.
    Starts from difference values between played drum-guitar onsets (currently in diff_hat)

    Parameters for determining diff:
        - style = 'constant' or 'diff' (does nothing) or 'EMA' (tba)
        - value = if None, will be computed as avg(diff_hat) over the present seq
                  if style='constant', diff   = value
                  if style='EMA',      EMA period = value
    Computes Y = (Y_hat + diff_hat - diff), in order to achieve ->
        -> a constant value for diff (see above)
    """
    if style == 'diff':
        Y = torch.Tensor(diff_hat)
        return Y_hat, Y
    diff = np.zeros_like(diff_hat)
    Y = np.zeros_like(Y_hat)
    for i in range(len(diff)):
        seq_len = int(X_lengths[i])
        if style == 'constant':
            if value:
                diff[i, :seq_len] = value
            else:  # default
                diff[i, :seq_len] = np.average(diff_hat[i][:seq_len])
        else:  # EMA TODO
            diff[i, :seq_len] = 0
        # Y = Y_hat + diff_hat - diff
        np.add(Y_hat[i], diff_hat[i], Y_hat[i])   # Y_hat = Y_hat + diff_hat
        np.subtract(Y_hat[i], diff[i], Y[i])      # Y     = Y_hat - diff

    Y = torch.Tensor(Y)
    Y_hat = torch.Tensor(Y_hat)

    return Y_hat, Y


def save_XY(X, X_lengths, diff_hat, Y, filename=None):
    """
    Save X, diff_hat, Y to a csv file.
    Returns total numbers of rows written.
    """
    Xcsv = X.numpy()
    Ycsv = Y.numpy()
    fmt = '%i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %g, %g, %g, %g, %g, %g, %g'
    header = "seq no., kick, snar, hclos, hopen, ltom, mtom, htom, cras, ride, duration, tempo, timesig, pos_in_bar, guitar, d_g_diff, y"
    columns = 1 + feat_vec_size + 1 + 1  # seq, fv, diff, y
    rows = int(sum(X_lengths))
    to_csv = np.zeros((rows, columns))
    cur_row = 0
    for i, x_len in enumerate(X_lengths):
        seq_len = int(X_lengths[i])
        for j in range(seq_len):
            to_csv[cur_row][0] = i
            to_csv[cur_row][1:feat_vec_size + 1] = Xcsv[i][j]
            to_csv[cur_row][feat_vec_size + 1] = diff_hat[i][j]
            to_csv[cur_row][feat_vec_size + 2] = Ycsv[i][j]
            cur_row += 1
    now = datetime.datetime.now()
    if filename == None:
        filename = "data/takes/" + now.strftime("%Y%m%d%H%M%S") + ".csv"
    np.savetxt(filename, to_csv, fmt=fmt, header=header)
    return cur_row


def load_XY(filename):
    """
    Get (unpadded) X, diff_hat, Y from a csv file.
    """
    X = np.zeros((1000, 64, feat_vec_size))  # seqs * hits * features
    Y = np.zeros((1000, 64))                 # seqs * hits
    diff_hat = np.zeros((1000, 64))          # seqs * hits
    X_lengths = np.zeros(1000)
    s_i = -1
    h_i = 0

    from_csv = np.loadtxt(filename, delimiter=',')
    for cur_row in range(len(from_csv)):
        cur_seq = int(from_csv[cur_row][0])
        if (s_i != cur_seq):
            # new seq
            X_lengths[s_i] = h_i
            s_i = cur_seq
            h_i = 0
        X[s_i][h_i] = from_csv[cur_row][1:feat_vec_size + 1]
        diff_hat[s_i][h_i] = from_csv[cur_row][feat_vec_size + 1]
        Y[s_i][h_i] = from_csv[cur_row][feat_vec_size + 2]
        h_i += 1
    # last seq
    X_lengths[s_i] = h_i
    batch_size = s_i + 1
    if DEBUG:
        print("Done loading sequences of lengths: ", X_lengths[:batch_size])
    return X, X_lengths, diff_hat, Y, batch_size


# TIMING NETWORK CLASS
# ====================


class TimingLSTM(nn.Module):
    def __init__(self, nb_layers=1, nb_lstm_units=100, input_dim=14, batch_size=10):
        """
        batch_size: # of sequences in training batch
        """
        super(TimingLSTM, self).__init__()

        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.input_dim = input_dim
        self.batch_size = batch_size

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_layers,
            batch_first=True,
            dropout=0.5
        )

        self.hidden = self.init_hidden()
        # output layer which projects back to Y space
        self.hidden_to_y = nn.Linear(self.nb_lstm_units, 1)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden = torch.randn(self.nb_layers,
                             self.batch_size, self.nb_lstm_units)
        cell = torch.randn(self.nb_layers,
                           self.batch_size, self.nb_lstm_units)
        """
        if self.hparams.on_gpu:
            hidden = hidden.cuda()
            cell = cell.cuda()
        """
        return (hidden, cell)

    def forward(self, X, X_lengths):
        # DON'T reset the LSTM hidden state. We want the LSTM to treat
        # a new batch as a continuation of a sequence (?)
        # self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()
        # print("X ....", X.size())

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # doesn't make sense to sort seqs by length => we lose ONNX exportability..
        X = torch.nn.utils.rnn.pack_padded_sequence(
            X, X_lengths, batch_first=True, enforce_sorted=False)

        # print("hidden", self.hidden[0].size(), "\n====")

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # hack for padded max length sequences:
        # https://github.com/pytorch/pytorch/issues/1591#issuecomment-365834629
        if X.size(1) < seq_len:
            dummy_tensor = Variable(torch.zeros(
                batch_size, seq_len - X.size(1), self.nb_lstm_units))
            X = torch.cat([X, dummy_tensor], 1)

        # Project to output space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden_to_y(X)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, 1 output)
        X = X.view(batch_size, seq_len, 1)

        y_hat = X
        return y_hat

    def loss(self, Y_hat, Y, X_lengths):
        """
        flatten all the targets and predictions,
        eliminate outputs on padded elements,
        compute MSE loss
        """
        Y = Y.view(-1)              # flat target
        Y_hat = Y_hat.view(-1)      # flat inference

        # filter out all zero positions from Y
        mask = (Y != 0)
        nb_outputs = torch.sum(mask).item()

        # pick the values for Y_hat and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0])] * mask

        criterion = nn.MSELoss(reduction='sum')

        # compute MSE loss
        return (criterion(Y_hat, Y) / nb_outputs)


# TRAIN METHOD
# ============

def train(model, dataloaders, minibatch_size=10, epochs=1):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_hist = np.zeros(epochs)

    for t in range(epochs):
        # train loop. TODO add several epochs, w/ noise?
        # TODO shuffle batches (not minibatches!)
        print('Epoch', t + 1, "/", epochs)
        epoch_loss = div_loss = 0.
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            for _, sample in enumerate(dataloaders[phase]):
                # always _[0] because dataloader.batch_size=1 (see train_gmd.py)
                X = sample['X'][0].to(device)
                X_lengths = sample['X_lengths'][0].to(device)
                Y = sample['Y'][0].to(device)
                model.hidden = model.init_hidden()

                n_mb = int(np.ceil(X.shape[0] / minibatch_size))

                for mb_i in range(n_mb):
                    if DEBUG:
                        print("miniBatch", mb_i + 1, "/", n_mb)
                    # get minibatch indices
                    if (mb_i + 1) * minibatch_size < X.shape[0]:
                        end = (mb_i + 1) * minibatch_size
                    else:
                        # reached the end
                        end = X.shape[0]
                    indices = torch.LongTensor(
                        range(mb_i * minibatch_size, end))
                    mb_X = torch.index_select(X, 0, indices)
                    mb_Xl = torch.index_select(X_lengths, 0, indices)
                    mb_Y = torch.index_select(Y, 0, indices)

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        #print(mb_X.size(), mb_Xl.size(), mb_Y.size())
                        mb_Y_hat = model(mb_X, mb_Xl)
                        loss = model.loss(mb_Y_hat, mb_Y, mb_Xl)
                        epoch_loss += loss.item()
                        div_loss += 1
                        if DEBUG:
                            print("LOSS:", loss.item())
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # detach/repackage the hidden state in between batches
                    model.hidden[0].detach_()
                    model.hidden[1].detach_()

            epoch_loss = epoch_loss / div_loss
            print(phase, "loss:", epoch_loss)

            if phase == 'train':
                scheduler.step()
                train_hist[t] = epoch_loss

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('====\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
