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
if __name__ == '__main__':
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
        '--batch_size', type=int, default=512,
        help='Batch size: how many files/takes to process at a time.')
    parser.add_argument(
        '--window_size', type=int, default=64,
        help='Minibatch / window size: number of 2*bars to train on.')
    parser.add_argument(
        '--epochs', type=int, default=0,
        help='# of epochs to train. Zero means don\'t train.')
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
    new_index = prev_i = 0
    X = np.zeros((1000, 64, feat_vec_size))  # seqs * hits * features
    Y_hat = np.zeros((1000, 64))             # seqs * hits
    X_lengths = np.zeros(1000)
    s_i = 0
    h_i = -1

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
            # quantized pos_in_bar --- 1. becomes 0.
            featVec[12] = H[index] % 1.
            featVec[13] = V[new_index]  # TODO CHECK IF MAYBE BETTER ZERO??

            if new_index > 0:
                # use average offset of first & last note in PREVIOUS hit-group
                featVec[14] = (O[new_index - 1] + O[prev_i]) / 2

            X, Y_hat, h_i, s_i, X_lengths = timing.addRow(
                featVec, None, X, Y_hat, h_i, s_i, X_lengths)

            # move on to the next (group of) note(s)
            featVec = np.zeros(feat_vec_size)
            prev_i = new_index
            new_index = index + 1
    return X, Y_hat, s_i + 1, X_lengths


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
    X, Y_hat, take_size, X_lengths = parseHOVtoFV(hits, offsets, vels, drumtrack, pitch_class_map,
                                                  tempos, timesigs, positions_in_bar)

    X, X_lengths, Y_hat = timing.prepare_X(
        X, X_lengths, Y_hat, take_size)
    Y_hat, Y = timing.prepare_Y(
        X_lengths, X[:, :, 14], Y_hat, style='diff')

    return X, X_lengths, Y


class GMDdataset(Dataset):
    """
    GMD dataset class. See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, csv_file, root_dir, source='csv', transform=None):
        self.meta = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.source = source
        self.transform = transform

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
                x, xl, y = pm_to_XY(file_name)
                # if get_y_n("Save to csv? "):
                csv_filename = file_name[:-3] + 'csv'
                rows, _ = timing.save_XY(x, xl, y, filename=csv_filename)
                print("Saved", csv_filename, ": ", rows, "rows.")
            else:
                # load CSV file
                csv_filename = file_name[:-3] + 'csv'
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, xl, y, bs = timing.load_XY(csv_filename)
                if sum(xl):
                    # don't process empty files
                    x, xl, _ = timing.prepare_X(
                        x, xl, None, bs)        # no y_hat (hasn't been run thru model)
                    _, y = timing.prepare_Y(
                        xl, x[:, :, 14], y, style='diff')
                # print("Loaded", csv_filename, ": ", bs, "bars.")

            if sum(xl):
                if self.transform:
                    x, y = self.transform(x, y)
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
    xx = [batch[i]['X'][j]
          for i in range(len(batch)) for j in range(len(batch[i]['X']))]
    xl = torch.tensor([batch[i]['X_lengths'][j]
                       for i in range(len(batch))
                       for j in range(len(batch[i]['X_lengths']))])
    yy = [batch[i]['Y'][j]
          for i in range(len(batch)) for j in range(len(batch[i]['Y']))]

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
        input_dim=feat_vec_size,
        batch_size=args.window_size,
        seq2seq=args.seq2seq)

    ### Pre-load ###
    if args.load_model:
        trained_path = args.load_model
        model.load_state_dict(torch.load(
            trained_path, map_location=timing.device))
        print("Loaded pre-trained model weights from", trained_path)

    ### Training & eval ###
    if args.epochs:
        print("Start training for", args.epochs, "epochs...")

        trained_model, loss = timing.train(model, dl,
                                           minibatch_size=args.window_size,
                                           epochs=args.epochs)

    # Optimisation ###            see https://optuna.org/
    if args.optuna:
        def objective(trial):
            layers = 2  # trial.suggest_int('layers', 2, 3)
            lstm_units = 256  # trial.suggest_int('lstm_units', 50, 550)
            dropout = 0.3  # trial.suggest_uniform('dropout', 0.2, 0.6)
            bs = 64  # pow(2, trial.suggest_int('bs', 1, 7))
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-4)
            ep = 150  # trial.suggest_int('ep', 3, 20)

            model = timing.TimingLSTM(nb_layers=layers, nb_lstm_units=lstm_units,
                                      input_dim=feat_vec_size, batch_size=bs, dropout=dropout,
                                      seq2seq=args.seq2seq)
            trained_model, loss = timing.train(
                model, dl, lr=lr, minibatch_size=bs, epochs=ep)

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
        else:
            PATH = "models/gmd_LSTM.pt"
        torch.save(trained_model.state_dict(), PATH)
        print("Saved trained model to", PATH)

"""

models:
simple -    1e-3 bs 64 epochs ???
boots -     1e-4 bs 64 epochs 43 -- to beat 0.0000071667 0.000059781
s2s -       4e-4 bs 64 epochs63 --- 6.1688e-5

"""
