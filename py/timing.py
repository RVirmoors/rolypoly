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
Y = np.zeros((1000, 64))                 # seqs * hits
h_i = 0
s_i = -1
loss = 0
X_lengths = np.zeros(1000)


def inference(featVec):
    return 0


def addRow(featVec, y, loss):
    global newSeq, X, Y, h_i, s_i, X_lengths
    # if new bar, finish existing sequence and start a new one
    if featVec[12] <= X[s_i][h_i][12]:
        if s_i >= 0:  # s_i is init'd as -1, so first note doesn't trigger:
            X_lengths[s_i] = int(h_i + 1)
            print("saved bar #", s_i, ", contains ", X_lengths[s_i], "hits.")
        s_i += 1
        h_i = 0
        X[s_i][0] = featVec         # first hit in new seq
        Y[s_i][0] = y
    else:
        h_i += 1
        X[s_i][h_i] = featVec
        Y[s_i][h_i] = y


def prepare_X():
    global X, X_lengths, s_i
    longest_seq = int(max(X_lengths))
    batch_size = s_i
    print("longest: ", longest_seq, " | batch size: ", batch_size)
    padded_X = np.ones((batch_size, longest_seq, feat_vec_size))
    for i, x_len in enumerate(X_lengths):
        if i < batch_size:
            x_len = int(x_len)
            print("BBB", x_len)
            sequence = X[i]
            padded_X[i, 0:x_len - 1] = sequence[:x_len - 1]
    print(padded_X)


class timingLSTM(nn.Module):
    def __init__(self, nb_layers, nb_lstm_units=100, input_dim=13, batch_size=10):
        """
        batch_size: # of sequences in training batch
        """
        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.input_dim = input_dim
        self.batch_size = batch_size

        self.lstm

        # build actual NN
        self.__build_model()

    def __build_model(self):
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_layers,
            batch_first=True,
        )
        # output layer which projects back to tag space
        self.hidden_to_y = nn.Linear(self.nb_lstm_units, 1)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.hparams.nb_layers,
                               self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.hparams.nb_layers,
                               self.batch_size, self.nb_lstm_units)

        if self.hparams.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
