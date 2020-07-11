"""
Rolypoly timing model
2020 rvirmoors

TODO methods:
new performance
add row
preprocess dataset
train model
inference

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

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


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
data_loader = torch.utils.data.DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=True)

"""

feat_vec_size = len(ROLAND_DRUM_PITCH_CLASSES) + 4 + 1

X = np.zeros((1000, 64, feat_vec_size))  # seqs * hits * features
Y = np.zeros((1000, 64))             # seqs * hits
Y_hat = np.zeros((1000, 64))             # seqs * hits
h_i = 0
s_i = -1
loss = 0
X_lengths = np.zeros(1000)

# METHODS FOR BUILDING X, Y
# =========================


def addRow(featVec, y_hat, diff):
    global newSeq, X, Y, Y_hat, h_i, s_i, X_lengths
    #print("add y", y, "diff ", diff, "to X[", s_i, "][", h_i + 1, "]")
    # if new bar, finish existing sequence and start a new one
    if featVec[12] <= X[s_i][h_i][12]:
        if s_i >= 0:  # s_i is init'd as -1, so first note doesn't trigger:
            # move delay to first hit in new seq
            Y_hat[s_i + 1][0] = Y_hat[s_i][h_i + 1]
            # last hit plus one doesn't make sense
            Y_hat[s_i][h_i + 1] = 0
            X_lengths[s_i] = h_i + 1
            print("saved bar #", s_i, "w/", int(X_lengths[s_i]), "hits.")
            print("Y_hat for seq:", Y_hat[s_i][:int(X_lengths[s_i])])
            print("==========")
        s_i += 1
        h_i = 0
        X[s_i][0] = featVec         # first hit in new seq
        Y_hat[s_i][1] = y_hat       # delay for next hit
        Y[s_i][0] = diff            # correct delay for this hit
    else:
        h_i += 1
        X[s_i][h_i] = featVec       # this hit
        Y_hat[s_i][h_i + 1] = y_hat  # delay for next hit
        Y[s_i][h_i] = diff          # correct delay for this hit


def prepare_X():
    """
    Pad short sequences
    https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    """
    global X, X_lengths, s_i, Y, Y_hat
    longest_seq = int(max(X_lengths))
    batch_size = s_i       # number of sequences (TODO add the last one?)
    print("longest: ", longest_seq, " | batch size: ", batch_size)
    padded_X = np.zeros((batch_size, longest_seq, feat_vec_size))
    padded_Y = np.zeros((batch_size, longest_seq))
    padded_Y_hat = np.zeros_like(padded_Y)
    for i, x_len in enumerate(X_lengths):
        if i < batch_size:
            x_len = int(x_len)
            # print("i", i, "seq length", x_len)
            sequence = X[i]
            padded_X[i, 0:x_len] = sequence[:x_len]
            sequence = Y[i]
            padded_Y[i, 0:x_len] = sequence[:x_len]
            sequence = Y_hat[i]
            padded_Y_hat[i, 0:x_len] = sequence[:x_len]
    X = padded_X
    Y = padded_Y
    Y_hat = padded_Y_hat


def prepare_Y(style='constant', value=None):
    """
    Computes Y, the "correct" values to be used in the MSE loss function.
    Starts from difference values between drum-guitar onsets (currently in Y)

    Parameters for determining diff:
        - style = 'constant' or 'diff' (does nothing) or 'EMA' (tba)
        - value = if None, will be computed as avg(diff_hat) over the present seq
                  if style='constant', diff   = value
                  if style='EMA',      EMA period = value
    Returns (Y_hat + diff_hat - diff), in order to achieve ->
        -> a constant value for diff (computed above)
    """
    if style == 'diff':
        return
    global X, X_lengths, Y, Y_hat
    diff = np.zeros_like(Y)
    for i in range(len(diff)):
        seq_len = int(X_lengths[i])
        if style == 'constant':
            if value:
                diff[i, :seq_len] = value
            else:  # default
                diff[i, :seq_len] = np.average(Y[i][:seq_len])
        else:  # EMA TODO
            diff[i, :seq_len] = 0
        # Y_hat = Y + diff - diff_hat
        np.add(Y_hat[i], Y[i], Y_hat[i])          # Y_hat = Y_hat + diff_hat
        np.subtract(Y_hat[i], diff[i], Y[i])      # Y = Y_hat - diff
    #print("Y_hat played:", Y_hat)
    #print("Y 'correct':", Y)

# TIMING NETWORK CLASS
# ====================


class TimingLSTM(nn.Module):
    def __init__(self, nb_layers=1, nb_lstm_units=100, input_dim=13, batch_size=10):
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
        )

        self.hidden = self.init_hidden()
        # output layer which projects back to tag space
        self.hidden_to_y = nn.Linear(self.nb_lstm_units, 1)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_layers,
                               self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_layers,
                               self.batch_size, self.nb_lstm_units)
        """
        if self.hparams.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()
        """

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths):
        # DON'T reset the LSTM hidden state. We want the LSTM to treat
        # a new batch as a continuation of a sequence (?)
        # self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(
            X, X_lengths, batch_first=True)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden_to_y(X)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, 1 output)
        X = X.view(batch_size, seq_len, 1)

        Y_hat = X
        return Y_hat

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
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

        # compute MSE loss
        return (nn.MSELoss(Y_hat, Y, reduction='sum') / nb_outputs)
