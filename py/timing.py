"""
Rolypoly timing model
2020 rvirmoors
"""
DEBUG = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# see https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import datetime
import time
import os
import copy

from constants import ROLAND_DRUM_PITCH_CLASSES
from helper import get_y_n, EarlyStopping, plot_grad_flow, ewma, roll_w_padding
import matplotlib.pyplot as plt
import random
from tqdm import tqdm, trange

# Deterministic results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

torch.set_printoptions(sci_mode=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feat_vec_size = len(ROLAND_DRUM_PITCH_CLASSES) + 6
# FV: 9 drumhits + dur, tempo, timesig, pos_in_bar, guitar, d_g-diff

# METHODS FOR BUILDING X, Y
# =========================


def addRow(featVec, y_hat, X, Y_hat, h_i, s_i, X_lengths, pre=False):
    # if new bar, finish existing sequence and start a new one
    if featVec[12] <= X[s_i][h_i][12] and h_i >= 0:
        # print("new bar", s_i, h_i)
        if DEBUG and not pre:
            print("added bar #", s_i, "w/", int(X_lengths[s_i]), "hits.")
            print("Y_hat for seq:", Y_hat[s_i][:int(X_lengths[s_i])])
            print("==========")
        s_i += 1
        h_i = 0
        X[s_i][0] = featVec          # first hit in new seq
        if pre:
            X_lengths[s_i] = 1
        else:
            Y_hat[s_i][0] = y_hat        # delay for next hit
    else:
        h_i += 1
        X[s_i][h_i] = featVec           # this hit
        if pre:
            X_lengths[s_i] = h_i + 1
        else:
            Y_hat[s_i][h_i] = y_hat     # delay for next hit
    # print(s_i, h_i, X[s_i][h_i][:9], X[s_i][h_i][12], X[s_i][h_i][14])
    return X, Y_hat, h_i, s_i, X_lengths


def prepare_X(X, X_lengths, Y_hat, batch_size):
    """
    Pad short sequences
    https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    """
    longest_seq = int(max(X_lengths))
    # if DEBUG:
    #    print("longest: ", longest_seq, " | batch size: ", batch_size)
    padded_X = np.zeros((batch_size, longest_seq, feat_vec_size))
    if Y_hat is not None:
        padded_Y_hat = np.zeros((batch_size, longest_seq))
    for i, x_len in enumerate(X_lengths[:batch_size]):
        x_len = int(x_len)
        # print("i", i, "seq length", x_len)
        sequence = X[i]
        padded_X[i, 0:x_len] = sequence[:x_len]
        if Y_hat is not None:
            sequence = Y_hat[i]
            padded_Y_hat[i, 0:x_len] = sequence[:x_len]
    X = padded_X
    if Y_hat is not None:
        Y_hat = padded_Y_hat
    X_lengths = X_lengths[:batch_size]
    X = torch.tensor(X, dtype=torch.float64)
    X_lengths = torch.tensor(X_lengths, dtype=torch.int64)
    return X, X_lengths, Y_hat


def prepare_Y(X_lengths, diff_hat, Y_hat, style='constant', value=None, online=False):
    """
    Computes Y, the target values to be used in the MSE loss function.
    Starts from difference values between played drum-guitar onsets (currently in diff_hat)

    Parameters for determining delta:
        - style = 'constant'                -> we want to bring next d-g delay close to avg
                or 'EMA'                    -> we want to bring next d-g delay close to EMA
                or 'diff' (rolls diff_hat)  -> we want to minimize next drum-guitar delay
        - value = if None, will be computed as avg(diff_hat) over the present seq
                  if style='constant', value -> delta
                  if style='EMA',      value -> EMA alpha in (0,1)

    (for style == 'constant' or 'EMA')
    Computes Y = (Y_hat + diff_hat[t+1] - delta), in order to achieve ->
        -> a constant value for diff_hat (see above)
    """

    if style == 'diff':
        if online:
            # try to predict the next d_g delay (here a single value)
            Y = diff_hat
            Y = torch.Tensor([Y]).double()
        else:
            # try to predict the next d_g delay
            # Y = torch.roll(diff_hat, -1) ## doesn't work due to padding!!! workaround:
            Y = roll_w_padding(diff_hat, X_lengths)
            Y_hat = torch.Tensor(Y_hat).double()  # dtype=torch.float64)
        return Y_hat, Y

    if torch.is_tensor(diff_hat):
        diff_hat = roll_w_padding(diff_hat, X_lengths).numpy()
    else:
        diff_hat = roll_w_padding(torch.tensor(
            diff_hat).double(), X_lengths).numpy()
    delta = np.zeros_like(diff_hat)
    Y = np.zeros_like(Y_hat)

    if style == 'constant':
        for i in range(len(delta)):
            seq_len = int(X_lengths[i])
            if value is not None:
                delta[i, :seq_len] = value
            else:  # default
                delta[i, :seq_len] = np.average(diff_hat[i][:seq_len])
    elif style == 'EMA':
        for i in range(len(diff_hat)):
            seq_len = int(X_lengths[i])
            delta[i, :seq_len] = ewma(
                diff_hat[i, :seq_len],
                alpha=value if value else 0.8)

    # Y[t] = Y_hat[t] + diff_hat[t+1] - diff[t+1]
    if style == 'constant' or 'EMA':
        np.add(Y_hat, diff_hat, Y_hat)   # Y_hat = Y_hat + diff_hat
        np.subtract(Y_hat, delta, Y)      # Y     = Y_hat - delta

    if style == 'KL':


    Y = torch.Tensor(Y).double()  # dtype=torch.float64)
    Y_hat = torch.Tensor(Y_hat).double()  # dtype=torch.float64)

    print("delta:", delta[:, 0])

    return Y_hat, Y


def save_XY(X, X_lengths, Y, Y_hat=None, filename=None):
    """
    Save X, Y to a csv file.
    Returns total numbers of rows written.
    """
    Xcsv = X.numpy()
    Ycsv = Y.numpy()
    if Y_hat is not None:
        Y_hcsv = Y_hat.numpy()
        fmt = '%i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %g, %g, %g, %g, %g, %g, %g, %g'
        header = "seq no., kick, snar, hclos, hopen, ltom, mtom, htom, cras, ride, duration, tempo, timesig, pos_in_bar, guitar, d_g_diff, y, y_hat"
    else:
        fmt = '%i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %g, %g, %g, %g, %g, %g, %g'
        header = "seq no., kick, snar, hclos, hopen, ltom, mtom, htom, cras, ride, duration, tempo, timesig, pos_in_bar, guitar, d_g_diff, y"
    columns = 1 + feat_vec_size + 1 + 1 * \
        (Y_hat is not None)  # seq, fv, y, y_hat
    rows = int(sum(X_lengths))
    to_csv = np.zeros((rows, columns))
    cur_row = s_i = 0
    for i, x_len in enumerate(X_lengths):
        seq_len = int(X_lengths[i])
        if seq_len:
            for j in range(seq_len):
                to_csv[cur_row][0] = s_i
                to_csv[cur_row][1:feat_vec_size + 1] = Xcsv[i][j]
                to_csv[cur_row][feat_vec_size + 1] = Ycsv[i][j]
                if Y_hat is not None:
                    to_csv[cur_row][feat_vec_size + 2] = Y_hcsv[i][j]
                cur_row += 1
            s_i += 1
    now = datetime.datetime.now()
    if filename == None:
        filename = "data/takes/" + now.strftime("%Y%m%d%H%M%S") + ".csv"
    np.savetxt(filename, to_csv, fmt=fmt, header=header)
    np.savetxt("data/takes/last.csv", to_csv, fmt=fmt, header=header)
    return cur_row, filename


def load_XY(filename):
    """
    Get (unpadded) X, Y from a csv file.
    """
    X = np.zeros((1000, 64, feat_vec_size))  # seqs * hits * features
    Y = np.zeros((1000, 64))                 # seqs * hits
    X_lengths = np.zeros(1000)
    s_i = 0
    h_i = 0
    batch_size = 0

    from_csv = np.loadtxt(filename, delimiter=',')
    if from_csv.ndim == 2:
        for cur_row in range(len(from_csv)):
            cur_seq = int(from_csv[cur_row][0])
            if (s_i != cur_seq and h_i):
                # new seq
                X_lengths[s_i] = h_i
                s_i = cur_seq
                h_i = 0
            X[s_i][h_i] = from_csv[cur_row][1:feat_vec_size + 1]
            Y[s_i][h_i] = from_csv[cur_row][feat_vec_size + 1]
            h_i += 1
        # last seq
        if h_i:
            X_lengths[s_i] = h_i
            batch_size = s_i + 1
        if DEBUG:
            print("Done loading sequences of lengths: ",
                  X_lengths[:batch_size])
    return X, X_lengths, Y, batch_size


def transform(X, Y):
    """
    TODO Data preprocessing:
    Drum hits (x[0:8]) and pos_in_bar (x[12]) rescaled from [0,1] to [-1,1]
    Minmax scale factor applied to diff_hat (x[14]) and y, y_hat to [-1,1]
    Duration (x[9]) rescaled from [20, 1000] to [-1,1]  (log, capped)
    Tempo (x[10]) rescaled from [60,240] to [-1,1]      (log, capped)
    Timesig (x[11]) rescaled from [0.25, 4] to [-1,1]   (log, capped)
    """


# TIMING NETWORK CLASS
# ====================


class TimingLSTM(nn.Module):
    def __init__(self, nb_layers=2, nb_lstm_units=256, input_dim=15, batch_size=64, dropout=0.3, seq2seq=False):
        """
        batch_size: # of sequences (bars) in training batch
        """
        super(TimingLSTM, self).__init__()

        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.dropout = dropout
        self.seq2seq = seq2seq

        self.init_hidden()

        if not self.seq2seq:
            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.nb_lstm_units,
                num_layers=self.nb_layers,
                batch_first=True,
                dropout=self.dropout
            ).double().to(device)

        else:
            # encoder is bidirectional LSTM
            self.encoder = nn.LSTM(
                input_size=self.input_dim - 2,  # no guitar: we don't know descr & g_d
                hidden_size=self.nb_lstm_units,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ).double().to(device)

            # decoder is 2-layer LSTM
            self.decoder = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.nb_lstm_units,
                num_layers=2,
                batch_first=True,
                dropout=self.dropout
            ).double().to(device)

        # tanh activation
        self.tanh = nn.Tanh()

        # output layer which projects back to Y space
        self.hidden_to_y = nn.Linear(self.nb_lstm_units, 1).double().to(device)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden = torch.zeros(self.nb_layers,
                             self.batch_size, self.nb_lstm_units, device=device, dtype=torch.float64)
        cell = torch.zeros(self.nb_layers,
                           self.batch_size, self.nb_lstm_units, device=device, dtype=torch.float64)
        self.hidden = (hidden, cell)

    def hidden_detach(self):
        # detach/repackage the hidden state in between batches
        self.hidden[0].detach_()
        self.hidden[1].detach_()

    def forward(self, X, X_lengths):
        # DON'T reset the LSTM hidden state. We DO want the LSTM to treat
        # a new batch as a continuation of a sequence!
        # self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()
        #print("X ....", X.size())
        # print(X_lengths)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # doesn't make sense to sort seqs by length => we lose ONNX exportability.
        X_pack = torch.nn.utils.rnn.pack_padded_sequence(
            X, X_lengths, batch_first=True, enforce_sorted=False)
        if self.seq2seq:
            X_source = torch.nn.utils.rnn.pack_padded_sequence(
                X[:, :, :13], X_lengths, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        if not self.seq2seq:
            X_lstm, self.hidden = self.lstm(X_pack, self.hidden)
        else:
            _, self.hidden = self.encoder(X_source, self.hidden)
            X_lstm, self.hidden = self.decoder(X_pack, self.hidden)

        # undo the packing operation
        X_lstm, _ = torch.nn.utils.rnn.pad_packed_sequence(
            X_lstm, batch_first=True)

        X_lstm = self.tanh(X_lstm)

        # hack for padded max length sequences:
        # https://github.com/pytorch/pytorch/issues/1591#issuecomment-365834629
        if X_lstm.size(1) < seq_len:
            dummy_tensor = torch.zeros(
                batch_size, seq_len - X_lstm.size(1), self.nb_lstm_units, device=device, dtype=torch.float64)
            X_lstm = torch.cat([X_lstm, dummy_tensor], 1)

        # Project to output space (Linear decoder)

        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        # First we need to reshape the data so it goes into the linear layer
        X_lstm = X_lstm.contiguous()
        X_lstm = X_lstm.view(-1, X_lstm.shape[2])
        # run through actual linear layer
        X_out = self.hidden_to_y(X_lstm)
        # Then back to (batch_size, seq_len, 1 output)
        X_out = X_out.view(batch_size, seq_len, 1)

        y_hat = X_out  # [-1][(X_lengths[-1] - 1)][0]
        return y_hat

    def loss(self, Y_hat, Y, diff_hat):
        """
        flatten all the targets and predictions,
        eliminate outputs on padded elements,
        compute MSE loss
        """
        Y = Y.view(-1)                  # flat target
        Y_hat = Y_hat.view(-1)          # flat inference
        if diff_hat is not None:
            diff_hat = diff_hat.view(-1)    # flat drum-guitar (for mask)

        # filter out all zero positions from Y
        mask = (Y != 0)

        # filter out repeated diff_hat
        if diff_hat is not None:
            diffMask = torch.BoolTensor([True]).to(
                device)  # first is always new
            b = [(diff_hat[i + 1] - diff_hat[i] != 0)
                 for i in range(diff_hat.shape[0] - 1)]
            diffMask = torch.cat(
                (diffMask, torch.BoolTensor(b).to(device)), dim=0)

            mask = diffMask * mask

        nb_outputs = torch.sum(mask).item()
        if nb_outputs == 0:
            nb_outputs = 1
        if DEBUG:
            print("Computing loss for", nb_outputs, "hits.")

        # pick the values for Y_hat and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0])] * mask
        Y = Y[range(Y.shape[0])] * mask

        criterion = nn.MSELoss(reduction='sum')

        # compute MSE loss
        return (criterion(Y_hat, Y) / nb_outputs)


# TRAIN METHOD
# ============

def train(model, dataloaders, minibatch_size=64, epochs=20, lr=1e-3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.

    model.to(device)
    print(model)
    print("window size:", minibatch_size, "bars | lr:", lr)
    print("Running on", next(model.parameters()).device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    writer = SummaryWriter()
    w_i = {'train': 0, 'val': 0}

    es = EarlyStopping(patience=10)
    early_stop = False

    if 'val' in dataloaders:
        phases = ['train', 'val']
    else:
        phases = ['train']

    for t in range(epochs):
        # train loop. TODO add noise?
        # print("Epoch", t + 1, "/", epochs)
        epoch_loss = div_loss = 0.
        if DEBUG:
            plt.ion()
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # (tqdm(dataloaders[phase], postfix={'phase': phase[0]})):
            for b_i, sample in enumerate(dataloaders[phase]):
                if sample['X'].shape[0] == 1:
                    sample['X'] = sample['X'].squeeze()
                    sample['X_lengths'] = sample['X_lengths'].squeeze()
                    sample['Y'] = sample['Y'].squeeze()
                X = sample['X'].to(device)
                X_lengths = sample['X_lengths'].to(device)
                Y = sample['Y'].to(device)
                model.init_hidden()  # reset the state at the start of a take

                n_mb = int(np.ceil(X.shape[0] / minibatch_size))

                batch_loss = 0
                batch_div = 0

                for mb_i in trange(n_mb):
                    # if DEBUG:
                    #    print("miniBatch", mb_i + 1, "/", n_mb)
                    # get minibatch indices
                    if (mb_i + 1) * minibatch_size < X.shape[0]:
                        end = (mb_i + 1) * minibatch_size
                    else:
                        # reached the end
                        end = X.shape[0]
                    indices = torch.tensor(
                        range(mb_i * minibatch_size, end), device=device, dtype=torch.int64)
                    mb_X = torch.index_select(X, 0, indices).to(device)
                    mb_Xl = torch.index_select(
                        X_lengths, 0, indices).to(device)
                    mb_Y = torch.index_select(Y, 0, indices).to(device)

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # print(mb_X.size(), mb_Xl.size(), mb_Y.size())
                        mb_Y_hat = model(mb_X, mb_Xl)
                        loss = model.loss(mb_Y_hat, mb_Y, mb_X[:, :, 14])
                        epoch_loss += loss.item()
                        batch_loss += loss.item()
                        div_loss += 1
                        batch_div += 1
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1)  # clip gradients
                            optimizer.step()

                    # detach/repackage the hidden state in between batches
                    model.hidden_detach()

                if DEBUG:
                    print(phase + 'Epoch: {} [Batch {}/{}]\t{:3d} seqs\tBatch loss: {:.6f}'.
                          format(t + 1, b_i + 1, len(dataloaders[phase]), n_mb, batch_loss / batch_div))

            epoch_loss = epoch_loss / div_loss
            if t % 1 == 0:
                print("Epoch", t + 1, phase, "loss:", epoch_loss)
            writer.add_scalar(
                "Loss/" + phase, epoch_loss, w_i[phase])
            w_i[phase] += 1

            if phase == 'train':
                scheduler.step()
                if DEBUG:
                    plot_grad_flow(model.named_parameters())
                    plt.pause(0.01)
            elif es.step(torch.tensor(epoch_loss)):
                print("Stopping early @ epoch", t + 1, "!")
                early_stop = True
                break

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_epoch = t + 1
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        if early_stop:
            print('Stopped.')
            break

    time_elapsed = time.time() - since
    print('====\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if 'val' in dataloaders:
        print('Best validation loss: {:4f}, found in Epoch #{:d}'.format(
            best_loss, best_epoch))
        print('Best validation MSE (16th note) loss: {:4f}'.format(
            best_loss * 16 * 16))
        # load best model weights
        model.load_state_dict(best_model_wts)

    if DEBUG:
        plt.show()

    ### Evaluation ###
    model.eval()
    total_loss = div_loss = 0

    # todo refactor this, too much copy-paste from above...
    if 'test' in dataloaders:
        for b_i, sample in enumerate(dataloaders['test']):
            if sample['X'].shape[0] == 1:
                sample['X'] = sample['X'].squeeze()
                sample['X_lengths'] = sample['X_lengths'].squeeze()
                sample['Y'] = sample['Y'].squeeze()
            X = sample['X'].to(device)
            X_lengths = sample['X_lengths'].to(device)
            Y = sample['Y'].to(device)

            model.init_hidden()

            n_mb = int(np.ceil(X.shape[0] / minibatch_size))

            batch_loss = 0
            batch_div = 0

            for mb_i in range(n_mb):
                # if DEBUG:
                #    print("miniBatch", mb_i + 1, "/", n_mb)
                # get minibatch indices
                if (mb_i + 1) * minibatch_size < X.shape[0]:
                    end = (mb_i + 1) * minibatch_size
                else:
                    # reached the end
                    end = X.shape[0]
                indices = torch.tensor(
                    range(mb_i * minibatch_size, end), device=device, dtype=torch.int64)
                mb_X = torch.index_select(X, 0, indices).to(device)
                mb_Xl = torch.index_select(
                    X_lengths, 0, indices).to(device)
                mb_Y = torch.index_select(Y, 0, indices).to(device)

                # forward, don't track history for eval
                with torch.set_grad_enabled(False):
                    # torch.zeros_like(mb_Y).to(device) # BASELINE ZERO
                    mb_Y_hat = model(mb_X, mb_Xl)
                    loss = model.loss(mb_Y_hat, mb_Y, None)
                    total_loss += loss.item()
                    batch_loss += loss.item()
                    div_loss += 1
                    batch_div += 1

            if DEBUG:
                print('Test: [Batch {}/{}]\t{:3d} seqs\tBatch loss: {:.6f}'.
                      format(b_i + 1, len(dataloaders['test']), n_mb, batch_loss / batch_div))

        total_loss = total_loss / div_loss
        print('Test loss: {:4f}'.format(total_loss))
        print('Test MSE (16th note) loss: {:4f}'.format(total_loss * 16 * 16))

    writer.add_hparams({'layers': model.nb_layers, 'lstm_units': model.nb_lstm_units, 'lr': lr, 'bsize': minibatch_size, 'epochs': epochs},
                       {'hparam/best_val_loss': best_loss, 'hparam/test_loss': total_loss})

    writer.flush()

    return model, total_loss


def trainOnline(model, y, y_hat, indices=-1, epochs=1, lr=1e-3):
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for t in range(epochs):
        model.train()  # Set model to training mode

        print(y[indices:])
        print(y_hat[indices:])

        y = y[indices:].to(device)
        y_hat = y_hat[indices:].to(device)  # already computed from model(x)

        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            loss = model.loss(y_hat, y, None)
            epoch_loss = loss.item()

            # backward + optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1)  # clip gradients
            optimizer.step()

            # detach/repackage the hidden state in between batches
            model.hidden_detach()

            scheduler.step()

    return model, epoch_loss
