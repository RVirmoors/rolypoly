"""Rolypoly Python implementation
2020 rvirmoors

Train network using Groove MIDI Dataset (GMD) from Magenta:
https://magenta.tensorflow.org/datasets/groove
"""


import pretty_midi
import numpy as np
import torch

import time
import os
import pandas as pd     # for quickly reading csv
from torch.utils.data import Dataset, DataLoader

import data
import timing           # ML timing module

from constants import ROLAND_DRUM_PITCH_CLASSES
from helper import get_y_n

np.set_printoptions(suppress=True)


feat_vec_size = timing.feat_vec_size


def quantizeDrumTrack(drumtrack, positions_in_bar, steps=16):
    """
    Start from live-played drumtrack. Return Hit, Offset, Velocity matrices.
    """
    live = positions_in_bar * steps
    H = live.round() / steps
    O = (live - live.round()) / steps
    V = np.zeros(len(drumtrack.notes))
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
    return X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths


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
    X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths = parseHOVtoFV(hits, offsets, vels, drumtrack, pitch_class_map,
                                                              tempos, timesigs, positions_in_bar)

    X, X_lengths, Y_hat, diff_hat = timing.prepare_X(
        X, X_lengths, Y_hat, diff_hat)
    Y_hat, Y = timing.prepare_Y(
        X_lengths, diff_hat, Y_hat, Y, style='diff')
    # if get_y_n("Save to csv? "):

    csv_filename = file_name[:-3] + 'csv'
    timing.save_XY(X, X_lengths, diff_hat, Y, filename=csv_filename)

    return X, X_lengths, Y


class GMDdataset(Dataset):
    """
    GMD dataset class. See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, csv_file, root_dir, source='csv'):
        self.meta = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.source = source

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = os.path.join(self.root_dir,
                                 self.meta.iloc[idx]['midi_filename'])
        if self.source == 'midi':
            # load MIDI file
            x, xl, y = pm_to_XY(file_name)
        else:
            # load CSV file
            csv_filename = file_name[:-3] + 'csv'
            x, xl, dh, y, bs = timing.load_XY(csv_filename)
            x, xl, yh, dh = timing.prepare_X(
                x, xl, dh, dh, bs)
            _, y = timing.prepare_Y(
                xl, dh, yh, y, style='diff')

        split = self.meta.iloc[idx]['split']

        return {'X': x, 'X_lengths': xl, 'Y': y, 'split': split}


# for i in range(len(gmd)):
#   sample = gmd[i]
# print(i, sample['pm'].time_signature_changes, sample['bpm'], sample['split'])

# print(gmd[432]['pm'].time_signature_changes,
#      gmd.meta.iloc[432]['time_signature'])


if __name__ == '__main__':
    # for now, just use batch_size = 1 because batches have different dimensions.
    # possible solutions:
    #   https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/17
    #   https://github.com/jihunchoi/recurrent-batch-normalization-pytorch
    #   https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/11
    #   https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    since = time.time()
    gmd = GMDdataset(csv_file='data/groove/info.csv',
                     root_dir='data/groove/')

    train_data = [gmd[i]
                  for i in range(len(gmd)) if gmd[i]['split'] == 'train']
    test_data = [gmd[i] for i in range(len(gmd)) if gmd[i]['split'] == 'test']
    val_data = [gmd[i]
                for i in range(len(gmd)) if gmd[i]['split'] == 'validation']

    dl = {}
    dl['train'] = DataLoader(train_data, batch_size=1,
                             shuffle=False, num_workers=1)
    dl['test'] = DataLoader(test_data, batch_size=1,
                            shuffle=False, num_workers=1)
    dl['val'] = DataLoader(val_data, batch_size=1,
                           shuffle=False, num_workers=1)

    time_elapsed = time.time() - since
    print('====\nData loaded in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(len(dl['train']), "training batches.",
          len(dl['val']), "val batches.")

    model = timing.TimingLSTM(
        input_dim=feat_vec_size, batch_size=len(dl['train']))

    print("Start training...")

    timing.train(model, dl, minibatch_size=1)
