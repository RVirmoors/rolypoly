"""Rolypoly Python implementation
2020 rvirmoors

Train network using Groove MIDI Dataset (GMD) from Magenta:
https://magenta.tensorflow.org/datasets/groove
"""
import argparse
import queue
import sys

import pretty_midi
import numpy as np
import torch
import optuna

import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import warnings
import pandas as pd     # for quickly reading csv
from torch.utils.data import Dataset, DataLoader

import data
import timing           # ML timing module

from constants import ROLAND_DRUM_PITCH_CLASSES
from helper import get_y_n

np.set_printoptions(suppress=True)
DEBUG = False

# parse command line args
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)  # show docstring from top
parser.add_argument(
    '--root_dir', default='data/groove/',
    help='Root directory for dataset.')
parser.add_argument(
    '--meta', default='info.csv',
    help='Metadata file: filename of csv list of samples for dataset.')
parser.add_argument(
    '--source', default='csv',
    help='Source data files: csv or midi.')
parser.add_argument(
    '--load_model', metavar='FOO.pt',
    help='Load a pre-trained model.')
parser.add_argument(
    '--batch_size', type=int, default=10,
    help='Batch size: how many files/takes to process at a time.')
parser.add_argument(
    '--window_size', type=int, default=128,
    help='Minibatch / window size: number of 2*bars to train on.')
parser.add_argument(
    '--hop_size', type=int, default=2,
    help='Training hop size.')
parser.add_argument(
    '--epochs', type=int, default=0,
    help='# of epochs to train. Zero means don\'t train.')
parser.add_argument(
    '--bootstrap', action='store_true',
    help='Bootstrap LSTM with position & guitar.')
parser.add_argument(
    '--seq2seq', action='store_true',
    help='Add LSTM decoder for a Seq2Seq model.')
parser.add_argument(
    '--optuna', action='store_true',
    help='Optimise (tune hyperparams) using Optuna.')
parser.add_argument(
    '--final', action='store_true',
    help='Final training, using the entire dataset.')
args = parser.parse_args()


feat_vec_size = timing.feat_vec_size


def quantizeDrumTrack(drumtrack, positions_in_bar, steps=16):
    """
    Start from live-played drumtrack. Return Hit, Offset, Velocity matrices.
    """
    live = positions_in_bar * steps
    H = live.round() / steps
    O = (live - live.round()) / steps
    V = np.zeros(len(drumtrack.notes))
    # check for triplets
    stepst = steps * 3 / 2
    livet = positions_in_bar * stepst
    Ht = livet.round() / stepst
    Ot = (livet - livet.round()) / stepst
    # find out indices of triplets (where the quant-diff is smaller)
    triplets = np.array(np.absolute(Ot) < np.absolute(O))
    H[triplets] = Ht[triplets]
    O[triplets] = Ot[triplets]
    return H, O, V


def parseHOVtoFV(H, O, V, drumtrack, pitch_class_map,
                 tempos, timesigs, positions_in_bar):
    featVec = np.zeros(feat_vec_size)  # 9+4+1 zeros
    new_index = 0
    X = np.zeros((1000, 64, feat_vec_size))  # seqs * hits * features
    Y = np.zeros((1000, 64))                 # seqs * hits
    Y_hat = np.zeros((1000, 64))             # seqs * hits
    diff_hat = np.zeros((1000, 64))          # seqs * hits
    X_lengths = np.zeros(1000)
    s_i = -1
    h_i = 0

    for index, note in enumerate(drumtrack.notes):
        if index < (len(drumtrack.notes) - 1):
            # if we're not at the last note, maybe wait
            currstart = H[index]        # quantized
            nextstart = H[index + 1]
            sleeptime = nextstart - currstart
            if sleeptime < 0:  # if end of bar...
                # ...then add barlength (1. for 4/4)
                sleeptime += timesigs[index][0] / timesigs[index][1]
        # one-hot encode feature vector [0...8]
        featVec[pitch_class_map[note.pitch]] = 1
        if sleeptime:
            # FV complete, process it
            featVec[9] = sleeptime * 1000.  # hit duration [ms]
            featVec[10] = tempos[new_index]  # use first hit in group of hits
            featVec[11] = timesigs[new_index][0] / timesigs[new_index][1]
            featVec[12] = positions_in_bar[new_index]
            featVec[13] = V[index]  # TODO CHECK IF MAYBE BETTER ZERO??

            # use average offset of first & last note in group
            y = (O[new_index] + O[index]) / 2

            X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths = timing.addRow(
                featVec, None, y, X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths)

            # move on to the next (group of) note(s)
            featVec = np.zeros(feat_vec_size)
            new_index = index + 1
    return X, Y, Y_hat, diff_hat, s_i + 1, X_lengths


def pm_to_XY(file_name):
    """
    Receives MIDI file name, returns X, Y
    """
    pm = pretty_midi.PrettyMIDI(file_name)

    if (len(pm.instruments) > 1):
        sys.exit('There are {} instruments. Please load a MIDI file with just one\
     (drum) instrument track.'.format(len(pm.instruments)))

    # vars
    drumtrack = pm.instruments[0]
    if (drumtrack.is_drum == False):
        sys.exit('Your MIDI file must be a DRUM track.')

    tc = pm.get_tempo_changes()
    ts = pm.time_signature_changes

    pitch_class_map = data.classes_to_map(ROLAND_DRUM_PITCH_CLASSES)
    tempos = data.score_tempo(drumtrack, tc)
    timesigs = data.score_timesig(drumtrack, ts)
    positions_in_bar = data.score_pos_in_bar(drumtrack, ts, tempos, timesigs)

    hits, offsets, vels = quantizeDrumTrack(drumtrack, positions_in_bar)
    X, Y, Y_hat, diff_hat, take_size, X_lengths = parseHOVtoFV(hits, offsets, vels, drumtrack, pitch_class_map,
                                                               tempos, timesigs, positions_in_bar)

    X, X_lengths, Y_hat, diff_hat = timing.prepare_X(
        X, X_lengths, Y_hat, diff_hat, take_size)
    Y_hat, Y = timing.prepare_Y(
        X_lengths, diff_hat, Y_hat, Y, style='diff')

    return X, X_lengths, diff_hat, Y


class GMDdataset(Dataset):
    """
    GMD dataset class. See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, csv_file, root_dir, source='csv'):
        self.meta = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.source = source

        self._remove_short_takes()  # filter out short samples

        self.x = [None] * len(self.meta)
        self.xl = [None] * len(self.meta)
        self.y = [None] * len(self.meta)
        self.split = [None] * len(self.meta)

        for idx in range(len(self.meta)):
            file_name = os.path.join(self.root_dir,
                                     self.meta.iloc[idx]['midi_filename'])
            if self.source == 'midi':
                # load MIDI file
                x, xl, dh, y = pm_to_XY(file_name)
                # if get_y_n("Save to csv? "):
                csv_filename = file_name[:-3] + 'csv'
                rows, _ = timing.save_XY(x, xl, dh, y, filename=csv_filename)
                print("Saved", csv_filename, ": ", rows, "rows.")
            else:
                # load CSV file
                csv_filename = file_name[:-3] + 'csv'
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, xl, dh, y, bs = timing.load_XY(csv_filename)
                if sum(xl):
                    # don't process empty files
                    x, xl, yh, dh = timing.prepare_X(
                        x, xl, dh, dh, bs)
                    _, y = timing.prepare_Y(
                        xl, dh, yh, y, style='diff')
                # print("Loaded", csv_filename, ": ", bs, "bars.")

            if sum(xl):
                self.x[idx] = x
                self.xl[idx] = xl
                self.y[idx] = y
                self.split[idx] = self.meta.iloc[idx]['split']
            else:  # not self.meta.iloc[idx]['bpm'].item():
                # drop one-bar takes that weren't dropped already
                self.split[idx] = 'dropped'
                # print("Dropped one-bar sample #", idx)

    def _remove_short_takes(self, min_dur=1):
        print("Filtering out samples shorter than", min_dur, "seconds...")
        old = len(self.meta)
        (self.meta).drop([i for i in range(len(self.meta))
                          if self.meta.iloc[i]['duration'] <= min_dur],
                         inplace=True)
        print("Dropped", old - len(self.meta), "short samples.")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {'fn': self.meta.iloc[idx]['midi_filename'], 'X': self.x[idx], 'X_lengths': self.xl[idx], 'Y': self.y[idx], 'split': self.split[idx]}


def pad_collate(batch):
    # used in dataLoader below
    xx = [torch.cat((batch[i]['X'][j], batch[i]['X'][j + 1]), dim=0)  # 2 bars at a time
          for i in range(len(batch))
          for j in range(len(batch[i]['X']) - 1)]
    xl = torch.tensor([batch[i]['X_lengths'][j]  # + batch[i]['X_lengths'][j + 1]
                       for i in range(len(batch))
                       for j in range(len(batch[i]['X_lengths']) - 1)])
    yy = [torch.cat((batch[i]['Y'][j], batch[i]['Y'][j + 1]), dim=0)
          for i in range(len(batch))
          for j in range(len(batch[i]['Y']) - 1)]

    take_lens = [len(x) for x in xx]

    xx_pad = torch.nn.utils.rnn.pad_sequence(
        xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(
        yy, batch_first=True, padding_value=0)

    return {'X': xx_pad, 'Y': yy_pad, 'X_lengths': xl, 'sample_lens': take_lens}


# https://pytorch.org/docs/1.1.0/_modules/torch/utils/data/dataloader.html
# https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
if __name__ == '__main__':
    since = time.time()
    gmd = GMDdataset(csv_file=args.root_dir + args.meta,
                     root_dir=args.root_dir,
                     source=args.source)

    train_data = [gmd[i]
                  for i in range(len(gmd)) if gmd[i]['split'] == 'train']
    test_data = [gmd[i] for i in range(len(gmd)) if gmd[i]['split'] == 'test']
    val_data = [gmd[i]
                for i in range(len(gmd)) if gmd[i]['split'] == 'validation']

    all_data = [gmd[i]
                for i in range(len(gmd)) if gmd[i]['split'] != 'dropped']

    dl = {}
    if args.final:
        dl['train'] = DataLoader(all_data, batch_size=args.batch_size,
                                 shuffle=True, num_workers=1, collate_fn=pad_collate)
    else:
        dl['train'] = DataLoader(train_data, batch_size=args.batch_size,
                                 shuffle=True, num_workers=1, collate_fn=pad_collate)
        dl['test'] = DataLoader(test_data, batch_size=args.batch_size,
                                shuffle=True, num_workers=1, collate_fn=pad_collate)
        dl['val'] = DataLoader(val_data, batch_size=args.batch_size,
                               shuffle=True, num_workers=1, collate_fn=pad_collate)

    time_elapsed = time.time() - since
    print('Data loaded in {:.0f}m {:.0f}s\n==========='.format(
        time_elapsed // 60, time_elapsed % 60))
    if args.final:
        print(len(dl['train']),
              "final training batches (", len(all_data), "files).")
    else:
        print(len(dl['train']), "training batches (", len(train_data), "files ).",
              len(dl['val']), "val batches (", len(val_data), "files ).")

    model = timing.TimingLSTM(
        input_dim=feat_vec_size, batch_size=args.window_size)

    ### Pre-load ###
    if args.load_model:
        trained_path = args.load_model
        model.load_state_dict(torch.load(trained_path))
        print("Loaded pre-trained model weights from", trained_path)

    ### Training & eval ###
    if args.epochs:
        print("Start training for", args.epochs, "epochs...")

        trained_model, loss = timing.train(model, dl,
                                           minibatch_size=args.window_size,
                                           minihop_size=args.hop_size,
                                           epochs=args.epochs,
                                           bootstrap=args.bootstrap,
                                           seq2seq=args.seq2seq)

    # Optimisation ###            see https://optuna.org/
    if args.optuna:
        def objective(trial):
            layers = 2  # trial.suggest_int('layers', 2, 3)
            lstm_units = trial.suggest_int('lstm_units', 50, 150)
            dropout = trial.suggest_uniform('dropout', 0.2, 0.6)
            bs = 256 # pow(2, trial.suggest_int('bs', 1, 7))
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            ep = 150  # trial.suggest_int('ep', 3, 20)

            model = timing.TimingLSTM(nb_layers=layers, nb_lstm_units=lstm_units,
                                      input_dim=feat_vec_size, batch_size=bs, dropout=dropout, 
                                      bootstrap=args.bootstrap, seq2seq=args.seq2seq)
            trained_model, loss = timing.train(
                model, dl, lr=lr, minibatch_size=bs, minihop_size=bs / 2, epochs=ep)

            return loss

        study = optuna.create_study(direction='minimize')
        # uses TPE Sampling: https://optuna.readthedocs.io/en/stable/reference/samplers.html#optuna.samplers.TPESampler
        study.optimize(objective, n_trials=20)

        print("Optimization done. Best params:", study.best_params)
        print("Best trial out of", len(study.trials), ":", study.best_trial)

        while get_y_n("Optimise for 10 more trials? "):
            study.optimize(objective, n_trials=10)

            print("Optimization done. Best params:", study.best_params)
            print("Best trial out of", len(study.trials), ":", study.best_trial)

    if get_y_n("Save trained model? "):
        if args.seq2seq:
            PATH = "models/gmd_seq2seq.pt"
        elif args.bootstrap:
            PATH = "models/gmd_LSTM_boot.pt"
        else:
            PATH = "models/gmd_LSTM.pt"
        torch.save(trained_model.state_dict(), PATH)
        print("Saved trained model to", PATH)

"""
[I 2020-07-30 19:49:15,228] Trial 0 finished with value: 5.4279108326188114e-05 and parameters: {'layers': 2, 'lstm_units': 96, 'dropout': 0.36116704798522165}. Best is trial 0 with value: 5.4279108326188114e-05.
[I 2020-07-30 20:06:48,166] Trial 1 finished with value: 5.6293804895667584e-05 and parameters: {'layers': 3, 'lstm_units': 76, 'dropout': 0.7655748051457263}. Best is trial 0 with value: 5.4279108326188114e-05.
[I 2020-07-30 20:20:08,448] Trial 2 finished with value: 5.521788244362895e-05 and parameters: {'layers': 2, 'lstm_units': 83, 'dropout': 0.3669492859191249}. Best is trial 0 with value: 5.4279108326188114e-05.
[I 2020-07-30 20:33:20,262] Trial 3 finished with value: 5.597054935721958e-05 and parameters: {'layers': 2, 'lstm_units': 76, 'dropout': 0.2787769789928112}. Best is trial 0 with value: 5.4279108326188114e-05.
[I 2020-07-30 20:46:54,300] Trial 4 finished with value: 5.614888685529846e-05 and parameters: {'layers': 2, 'lstm_units': 113, 'dropout': 0.5025545093135102}. Best is trial 0 with value: 5.4279108326188114e-05.
[I 2020-07-30 21:00:44,023] Trial 5 finished with value: 5.621860841961644e-05 and parameters: {'layers': 2, 'lstm_units': 131, 'dropout': 0.21875945580556236}. Best is trial 0 with value: 5.4279108326188114e-05.
[I 2020-07-30 21:19:15,729] Trial 6 finished with value: 5.405071509594314e-05 and parameters: {'layers': 3, 'lstm_units': 144, 'dropout': 0.29038152405608725}. Best is trial 6 with value: 5.405071509594314e-05.
[I 2020-07-30 21:36:36,791] Trial 7 finished with value: 5.6443440071678786e-05 and parameters: {'layers': 3, 'lstm_units': 55, 'dropout': 0.6460760596875088}. Best is trial 6 with value: 5.405071509594314e-05.
[I 2020-07-30 21:50:16,623] Trial 8 finished with value: 5.458007610872348e-05 and parameters: {'layers': 2, 'lstm_units': 119, 'dropout': 0.6986539530356477}. Best is trial 6 with value: 5.405071509594314e-05.
[I 2020-07-30 22:04:06,334] Trial 9 finished with value: 5.5564957211898746e-05 and parameters: {'layers': 2, 'lstm_units': 129, 'dropout': 0.3438758806117629}. Best is trial 6 with value: 5.405071509594314e-05.
[I 2020-07-30 22:22:42,517] Trial 10 finished with value: 5.558185038321966e-05 and parameters: {'layers': 3, 'lstm_units': 135, 'dropout': 0.4844698415965379}. Best is trial 6 with value: 5.405071509594314e-05.
[I 2020-07-30 22:40:35,044] Trial 11 finished with value: 5.544606147685907e-05 and parameters: {'layers': 3, 'lstm_units': 98, 'dropout': 0.4220523212604985}. Best is trial 6 with value: 5.405071509594314e-05.
"""
