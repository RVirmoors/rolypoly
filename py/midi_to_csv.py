"""Rolypoly Python implementation
2023 rvirmoors

Export midi drum performances (from GMD or elsewhere) to csv files.
"""

import torch

import pretty_midi
import numpy as np
import pandas as pd
import os

import data # data helper methods
import constants

from absl import app
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string("root_dir", 'data/groove', "Root directory for dataset.")
flags.DEFINE_string("meta", 'info.csv', 
    "Meta data file: csv list of samples for datset.")

# === DATASET FUNCTIONS ===

def removeShortTakes(meta, min_dur=5):
    print("Filtering out samples shorter than", min_dur, "seconds...")
    old = len(meta)
    (meta).drop([i for i in range(len(meta))
                        if meta.iloc[i]['duration'] <= min_dur],
                        inplace=True)
    print("Dropped", old - len(meta), "short samples.")
    return meta

def midifileToX_take(filename: str) -> torch.Tensor:
    """
    Convert a midi file to a tensor of X_decoder feature vectors.
    input: path to midi file
    output: tensor (len, constants.X_DECODER_CHANNELS)
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
    positions_in_bar = data.score_bar_pos(drumtrack, ts, tempos, timesigs)

    hits, offsets = quantizeDrumTrack(positions_in_bar)
    X_take = parseHO(drumtrack, pitch_class_map, tempos, timesigs, hits, offsets)
    
    return X_take

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
    output: X_take (len, constants.X_DECODER_CHANNELS)
    """
    hit_index = 0
    hit = torch.zeros(9)
    off = torch.zeros(9)
    pos = torch.zeros(3)

    for index, note in enumerate(drumtrack.notes):
        # print("index", index, "hit_index", hit_index, ":", note.pitch, "- offset:", O[index])
        if index < len(drumtrack.notes) - 1:
            duration = H[index + 1] - H[index]
            if duration < 0: # if at end of bar, then add barlength
                duration += timesigs[index][0] / timesigs[index][1]
        hit[pitch_class_map[note.pitch]] = note.velocity
        off[pitch_class_map[note.pitch]] = O[index]
        if duration:
            # done adding notes at this timestep, process it
            pos[0] = tempos[index]
            pos[1] = timesigs[index][0] / timesigs[index][1]
            pos[2] = H[index] # bar position, [0 - # of bars]
            # tau_guitar remains zero

            # add hit to X_take
            new_row = torch.cat((hit.unsqueeze(0), off.unsqueeze(0), pos.unsqueeze(0)), dim=1)
            if hit_index == 0:
                X_take = new_row
            else:
                X_take = torch.cat((X_take, new_row), 0)
            hit_index += 1
            duration = 0
            hit = torch.zeros(9)
            off = torch.zeros(9)
            pos = torch.zeros(3)
    return X_take

# === MAIN ===

def main(argv):
    meta = pd.read_csv(os.path.join(FLAGS.root_dir, FLAGS.meta))
    meta = removeShortTakes(meta)

    for idx in range(len(meta)):
        file_name = os.path.join(FLAGS.root_dir,
                                        meta.iloc[idx]['midi_filename'])
        csv_filename = file_name[:-3] + 'csv'

        x_take = midifileToX_take(file_name)
        rows = data.saveTakeToCSV(x_take, filename=csv_filename)
        print("Saved", csv_filename, ": ", rows, "rows.")


if __name__ == '__main__':
    app.run(main)
    # csv_filename = "gmd.csv"
    # x_take = midifileToX_take("gmd.mid")
    # print(x_take[:5])
    # rows = data.saveTakeToCSV(x_take, filename=csv_filename)
    # print("Saved", csv_filename, ": ", rows, "rows.")