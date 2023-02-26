"""Rolypoly Python implementation
2023 rvirmoors

Train network using Groove MIDI Dataset (GMD) from Magenta:
https://magenta.tensorflow.org/datasets/groove

Code heavily inspired by https://github.com/karpathy/nanoGPT/blob/master/train.py
"""

import torch
from torch.utils.data import Dataset, DataLoader

import pretty_midi
import numpy as np
import pandas as pd
import os
import time

import data # data helper methods
import model # model definition
import train # training methods
from helper import get_y_n


from absl import app
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string("root_dir", 'data/groove', "Root directory for dataset.")
flags.DEFINE_string("meta", 'info.csv', 
    "Meta data file: csv list of samples for datset.")
flags.DEFINE_enum("source", 'csv', ['csv', 'midi'], "Source data files.")
flags.DEFINE_string("load_model", "out/ckpt.pt", "Load pre-trained model from file.")
flags.DEFINE_integer("batch_size", 512, 
    "Batch size: how many minibatches to process at a time.")
flags.DEFINE_integer("block_size", 16,
    "Block / minibatch size: how many notes to look at.")
flags.DEFINE_integer("epochs", 100, "Number of epochs to train.")
flags.DEFINE_bool("final", False, "Final training, using all data.")


# === GLOBALS ===
feat_vec_size = data.X_DECODER_CHANNELS
train_data = {}
val_data = {}
train_data['X_dec'] = [] # lists of tensors
train_data['X_enc'] = []
train_data['Y'] = []
val_data['X_dec'] = []
val_data['X_enc'] = []
val_data['Y'] = []

# === DATASET FUNCTIONS ===

def removeShortTakes(meta, min_dur=5):
    print("Filtering out samples shorter than", min_dur, "seconds...")
    old = len(meta)
    (meta).drop([i for i in range(len(meta))
                        if meta.iloc[i]['duration'] <= min_dur],
                        inplace=True)
    print("Dropped", old - len(meta), "short samples.")
    return meta

def smidifileToY(filename: str) -> torch.Tensor:
    """
    Convert a midi file to a tensor of X_decoder feature vectors.
    input: path to midi file
    output: tensor (len, feat_vec_size)
    """
    pm = pretty_midi.PrettyMIDI(filename)

    if (len(pm.instruments) > 1):
        sys.exit('There are {} instruments. Please load a MIDI file with just one\
            (drum) instrument track.'.format(len(pm.instruments)))

    # vars
    drumtrack = pm.instruments[0]
    if (drumtrack.is_drum == False):
        sys.exit('Your MIDI file must be a DRUM track.')

    tc = pm.get_tempo_changes()
    ts = pm.time_signature_changes

    pitch_class_map = data.classes_to_map()
    tempos = data.score_tempo(drumtrack, tc)
    timesigs = data.score_timesig(drumtrack, ts)
    positions_in_bar = data.score_pos_in_bar(drumtrack, ts, tempos, timesigs)

    hits, offsets = quantizeDrumTrack(positions_in_bar)
    X_dec = parseHO(drumtrack, pitch_class_map, tempos, timesigs, hits, offsets)
    
    return X_dec

def quantizeDrumTrack(positions_in_bar, steps=16):
    """
    Start from live-played drumtrack. Return Hit, Offset matrices.
    TODO: normalize offsets according to average offset?
    """
    live = positions_in_bar * steps
    H = live.round() / steps
    O = (live - live.round()) / steps
    # check for triplets
    stepst = steps * 3 / 2
    livet = positions_in_bar * stepst
    Ht = livet.round() / stepst
    Ot = (livet - livet.round()) / stepst
    # find out indices of triplets (where the quant-diff is smaller)
    triplets = np.array(np.absolute(Ot) < np.absolute(O))
    H[triplets] = Ht[triplets]
    O[triplets] = Ot[triplets]
    return H, O

def parseHO(drumtrack, pitch_class_map, tempos, timesigs, H, O) -> torch.Tensor:
    """
    input: drumtrack, pitch_class_map, tempos, timesigs, H, O
    output: X_dec (len, feat_vec_size)
    """
    hit_index = 0

    for index, note in enumerate(drumtrack.notes):
        hit = torch.zeros(feat_vec_size)
        if index < len(drumtrack.notes) - 1:
            duration = H[index + 1] - H[index]
            if duration < 0: # if at end of bar, then add barlength
                duration += timesigs[index][0] / timesigs[index][1]
        hit[pitch_class_map[note.pitch]] = note.velocity / 127
        if duration:
            # done adding notes at this timestep, process it
            hit[9] = tempos[hit_index]
            hit[10] = timesigs[hit_index][0] / timesigs[hit_index][1]
            hit[11] = H[index] % 1.     # position in bar, 0-1
            hit[12] = (O[index] + O[hit_index]) / 2 # tau_drums
            # hit[13] remains zero (tau_guitar)

            # add hit to X_dec
            if hit_index == 0:
                X_dec = hit.unsqueeze(0)
            else:
                X_dec = torch.cat((X_dec, hit.unsqueeze(0)), 0)
            hit_index += 1
            duration = 0
    return X_dec


# === MAIN ===

def main(argv):
    meta = pd.read_csv(os.path.join(FLAGS.root_dir, FLAGS.meta))
    meta = removeShortTakes(meta)

    for idx in range(15):#len(meta)):
        file_name = os.path.join(FLAGS.root_dir,
                                        meta.iloc[idx]['midi_filename'])
        csv_filename = file_name[:-3] + 'csv'
        if FLAGS.source == 'midi':
            y = smidifileToY(file_name)
            rows = data.saveYtoCSV(y, filename=csv_filename)
            print("Saved", csv_filename, ": ", rows, "rows.")
        elif FLAGS.source == 'csv':
            y = data.loadYFromCSV(csv_filename)
            print("Loaded", csv_filename, ": ", y.shape[0], "rows.")
            if (y.shape[0] <= FLAGS.block_size):
                print("Skipping", csv_filename, "because it's too short.")
                continue
            xd, xe = getXforTrainFromY(y, FLAGS.block_size)
            if FLAGS.final or meta.iloc[idx]['split'] == 'train':
                train_data['Y'].append(y)
            else:
                val_data['Y'].append(y)

    config = model.Config()
    config.block_size = FLAGS.block_size
    m = model.Transformer(config)

    train.train(m, config, FLAGS.load_model, FLAGS.epochs, train_data, val_data, FLAGS.batch_size)


if __name__ == '__main__':
    app.run(main)