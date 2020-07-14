"""Rolypoly Python implementation
2020 rvirmoors

Train network using Groove MIDI Dataset (GMD) from Magenta:
https://magenta.tensorflow.org/datasets/groove
"""


import pretty_midi
import numpy as np
import torch

import data
import timing           # ML timing module

from constants import ROLAND_DRUM_PITCH_CLASSES
from helper import get_y_n

np.set_printoptions(suppress=True)

# load MIDI file
pm = pretty_midi.PrettyMIDI(
    'data/groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid')

if (len(pm.instruments) > 1):
    sys.exit('There are {} instruments. Please load a MIDI file with just one\
 (drum) instrument track.'.format(len(pm.instruments)))

# vars
drumtrack = pm.instruments[0]
if (drumtrack.is_drum == False):
    sys.exit('Your MIDI file must be a DRUM track.')

feat_vec_size = timing.feat_vec_size
tc = pm.get_tempo_changes()
ts = pm.time_signature_changes

GUITAR = 0

pitch_class_map = data.classes_to_map(ROLAND_DRUM_PITCH_CLASSES)
tempos = data.score_tempo(drumtrack, tc)
timesigs = data.score_timesig(drumtrack, ts)
positions_in_bar = data.score_pos_in_bar(drumtrack, ts, tempos, timesigs)

# print(positions_in_bar)


def quantizeDrumTrack(steps=16):
    """
    Start from live-played drumtrack. Return Hit, Offset, Velocity matrices.
    """
    live = positions_in_bar * steps
    H = live.round() / steps
    O = (live - live.round()) / steps
    V = np.zeros(len(drumtrack.notes))
    return H, O, V


def parseHOVtoFV(H, O, V):
    featVec = np.zeros(feat_vec_size)  # 9+4+1 zeros
    new_index = 0
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
            # print("y:", y)
            timing.addRow(featVec, None, y)  # build X & d_g_riff

            # move on to the next (group of) note(s)
            featVec = np.zeros(feat_vec_size)
            new_index = index + 1


hits, offsets, vels = quantizeDrumTrack()
parseHOVtoFV(hits, offsets, vels)

timing.prepare_X()
timing.prepare_Y('diff')

if get_y_n("Save to csv? "):
    timing.save_XY()

timing.train(timing.s_i + 1)
