"""
Rolypoly timing-meta model
Inputs: var(diff_hat), hidden state of TimingLSTM
Outputs: A, B

2020 rvirmoors
"""
DEBUG = True
HIDDEN_DIM = 256

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import datetime
import time
import os
import copy
from tqdm import tqdm, trange
from helper import get_y_n, EarlyStopping

# Deterministic results
import random
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


def add_XY(X, Y, varDiff, hidden, A, B):
    Xadd = torch.cat((varDiff, hidden), 1)
    Yadd = torch.cat((A, B), 1)
    X = torch.cat((X, Xadd), 0)
    Y = torch.cat((Y, Yadd), 0)
    if DEBUG:
        print("add X:", X[-5:, :5])
        print("add Y:", Y[-5:])
    return X, Y


def save_XY(X, Y, filename=None):
    Xcsv = X.numpy()
    Ycsv = Y.numpy()
    rows = len(Xcsv)
    columns = 1 + HIDDEN_DIM + 2  # 1 + 256 + 2
    to_csv = np.zeros((rows, columns))
    to_csv[:, :HIDDEN_DIM + 1] = Xcsv
    to_csv[:, -2:] = Ycsv
    if DEBUG:
        print("to_csv:", to_csv[:5])

    header = 'varDiff, ' + 'h, ' * HIDDEN_DIM + 'A, B'

    now = datetime.datetime.now()
    if filename == None:
        filename = "data/meta/" + now.strftime("%Y%m%d%H%M%S") + ".csv"
    np.savetxt(filename, to_csv, header=header)
    np.savetxt("data/meta/last.csv", to_csv, header=header)

    print("Saved", filename, ":", rows, "rows.")
    return rows, filename


def load_XY(filename="data/meta/last.csv"):
    data = np.loadtxt(filename)
    X = torch.DoubleTensor(data[:, :HIDDEN_DIM + 1])
    Y = torch.DoubleTensor(data[:, -2:])
    if DEBUG:
        print("load X", X[-5:, :5])
        print("load Y", Y[-5:])
    return X, Y, X.size()[0]    # inputs, targets, batch_size

# META DATASET CLASS
# ==================


class MetaDataset(Dataset):
    """
    MetaDataset class. See
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
    """

    def __init__(self, csv_file="data/meta/last.csv", transform=None):
        self.csv_file = csv_file
        self.transform = transform

        self.x, self.y, self.len = load_XY(csv_file)
        if self.transform:
            self.x, self.y = self.transform(self.x, self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {'X': self.x[idx], 'Y': self.y[idx]}


# TIMING META CLASS
# =================


class TimingMeta(nn.Module):
    def __init__(self, nb_layers=2, nb_lstm_units=1024, input_dim=HIDDEN_DIM + 1, batch_size=64, dropout=0.3):
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
            self.nb_lstm_units, 2).double().to(device)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden = torch.zeros(self.nb_layers,
                             self.batch_size, self.nb_lstm_units, device=device, dtype=torch.float64)
        cell = torch.zeros(self.nb_layers,
                           self.batch_size, self.nb_lstm_units, device=device, dtype=torch.float64)
        # print(hidden.size())
        self.hidden = (hidden, cell)

    def hidden_detach(self):
        # detach/repackage the hidden state in between batches
        self.hidden[0].detach_()
        self.hidden[1].detach_()

    def forward(self, X):
        # DON'T reset the LSTM hidden state. We DO want the LSTM to treat
        # a new batch as a continuation of a sequence!
        # self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()
        # print("X ....", X.size())

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
        # Then back to (batch_size, seq_len, 2 outputs)
        X_out = X_out.view(batch_size, seq_len, 2)

        y_hat = X_out
        return y_hat

    def loss(self, Y_hat, Y):
        """
        flatten all the targets and predictions,
        eliminate outputs on padded elements,
        compute MSE loss
        """
        Y = Y.view(-1)                  # flat target
        Y_hat = Y_hat.view(-1)          # flat inference

        criterion = nn.MSELoss(reduction='sum')
        # compute MSE loss
        loss = criterion(Y_hat, Y)

        return loss

# TRAIN METHOD
# ============


def train(model, dataloaders, minibatch_size=1, epochs=20, lr=1e-3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.

    model.to(device)
    print(model)
    print("minibatch size:", minibatch_size, "bars | lr:", lr)
    print("Running on", next(model.parameters()).device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    es = EarlyStopping(patience=20)
    early_stop = False

    phases = ['train', 'val']

    for t in range(epochs):
        # train loop. TODO add noise?
        epoch_loss = div_loss = 0.

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            for b_i, sample in enumerate(tqdm(dataloaders[phase], postfix={'phase': phase[0]})):
                X = sample['X'].unsqueeze(dim=0).to(device)
                Y = sample['Y'].to(device)
                model.init_hidden()  # reset the state at the start of a take
                batch_loss = 0
                batch_div = 0

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # print(mb_X.size(), mb_Xl.size(), mb_Y.size())
                    Y_hat = model(X)
                    loss = model.loss(Y_hat, Y)
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

            epoch_loss = epoch_loss / div_loss
            if t % 5 == 1:
                print("Epoch", t + 1, phase, "loss:", epoch_loss)

            if phase == 'train':
                scheduler.step()
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

    print('Best validation loss: {:4f}, found in Epoch #{:d}'.format(
        best_loss, best_epoch))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_loss


"""
TEST


HIDDEN_DIM = 3
X = torch.arange(1 + HIDDEN_DIM).double().unsqueeze(dim=0)
Y = torch.DoubleTensor([[10., 20.]])

varDiff = torch.DoubleTensor([[0.9]])
hidden = torch.arange(HIDDEN_DIM).double().unsqueeze(dim=0)
print(varDiff.size())

A = torch.DoubleTensor([[11.]])
B = torch.DoubleTensor([[21.]])

X, Y = add_XY(X, Y, varDiff, hidden, A, B)

save_XY(X, Y)

X, Y, batch_size = load_XY()
"""

if __name__ == '__main__':
    since = time.time()

    dataset = MetaDataset()
    batch_size = 1  # TODO fix for more
    validation_split = .2
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    dl = {}
    dl['train'] = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=train_sampler)
    dl['val'] = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=valid_sampler)
    time_elapsed = time.time() - since
    print('Data loaded in {:.0f}m {:.0f}s\n==========='.format(
        time_elapsed // 60, time_elapsed % 60))

    print(len(dl['train']), "training batches.",
          len(dl['val']), "val batches.")

    model = TimingMeta(batch_size=batch_size)

    trained_model, loss = train(model, dl,
                                epochs=100)

    getNextAB = dataset[-1]['X']
    getNextAB[0] = 0.   # predict for (desired) zero diff variance
    nextAB = model(getNextAB.unsqueeze(dim=0).unsqueeze(dim=0).to(device))
    print("A and B should go towards:", nextAB)

    if get_y_n("Save trained model? "):
        PATH = "models/__meta__.pt"
        torch.save(trained_model.state_dict(), PATH)
        print("Saved trained model to", PATH)
