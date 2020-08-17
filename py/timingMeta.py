"""
Rolypoly timing-meta model
2020 rvirmoors
"""
DEBUG = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

# METHODS FOR BUILDING X, Y
# =========================


def build_XY(varDiff, hidden, A, B):
    return X, Y


def save_XY(X, Y):
    return rows, filename


def load_XY(filename):
    return X, Y, batch_size


# TIMING META CLASS
# =================


class TimingMeta(nn.Module):
    def __init__(self, nb_layers=2, nb_lstm_units=1024, input_dim=257, batch_size=64, dropout=0.3):
        """
        batch_size: # of sequences (bars) in training batch
        """
        super(TimingMeta, self).__init__()

        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.dropout = dropout

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_layers,
            batch_first=True,
            dropout=self.dropout
        ).double().to(device)

        # self.layerNorm = nn.LayerNorm(nb_lstm_units) # TODO try this before tanh
        self.tanh = nn.Tanh()

        # output layer which projects back to Y space
        self.hidden_to_y = nn.Linear(
            self.nb_lstm_units, 1).double().to(device)

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

        # now run through LSTM
        X_lstm, self.hidden = self.lstm(X, self.hidden)

        X_lstm = self.tanh(X_lstm)

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

    def loss(self, Y_hat, Y):
        """
        flatten all the targets and predictions,
        eliminate outputs on padded elements,
        compute MSE loss
        """
        Y = Y.view(-1)                  # flat target
        Y_hat = Y_hat.view(-1)          # flat inference

        # filter out all zero positions from Y
        mask = (Y != 0)

        nb_outputs = torch.sum(mask).item()
        if nb_outputs == 0:
            nb_outputs = 1

        # pick the values for Y_hat and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0])] * mask
        Y = Y[range(Y.shape[0])] * mask

        criterion = nn.MSELoss(reduction='sum')
        # compute MSE loss
        loss = criterion(Y_hat, Y) / nb_outputs

        return loss
