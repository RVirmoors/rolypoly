"""
Rolypoly Python implementation
2023 rvirmoors

data preprocessing methods
"""

import torch
import numpy as np

X_DECODER_CHANNELS = 13 # 9 drum channel velocities + bpm, tsig, pos_in_bar, tau
X_ENCODER_CHANNELS = 5 # guitar velocity + bpm, tsig, pos_in_bar, tau_guitar

# === HELPER FUNCTIONS ===

def classes_to_map():
    """
    Build a pitch class map. (Kicks are class 0 etc)
    """
    # modified from https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/data.py
    classes = [
        # kick drum
        [35, 36],
        # snare drum
        [38, 37, 40],
        # closed hi-hat
        [42, 22, 44],
        # open hi-hat
        [46, 26],
        # low tom
        [43, 58],
        # mid tom
        [47, 45],
        # high tom
        [50, 48],
        # crash cymbal
        [49, 52, 55, 57],
        # ride cymbal
        [51, 53, 59]
    ]
    class_map = torch.zeros(128, dtype=torch.int64)
    for cls, pitches in enumerate(classes):
        for pitch in pitches:
            class_map[pitch] = cls
    return class_map

def ms_to_bartime(ms, featVec):
    """
    Convert a ms time difference to a bar-relative diff.
    input:
        ms = time to be converted
        featVec = feature of the note we relate to
    """
    tempo = featVec[10]
    timeSig = featVec[11]
    barDiff = ms / 1000 * 60 / tempo / timeSig
    return barDiff


def bartime_to_ms(bartime, featVec):
    """
    Convert a bar-relative time difference to a ms interval.
    input:
        bartime = time to be converted
        featVec = feature of the note we relate to
    """
    tempo = featVec[10]
    timeSig = featVec[11]
    ms = bartime * 1000 / 60 * tempo * timeSig
    return ms

# === DATA PROCESSING ===

def readScore(input: torch.Tensor):
    # input: (batch, 5, vec_size) from cpp host
    # output: (batch, 12, vec_size)
    pitch_class_map = classes_to_map()
    X_score = torch.zeros(input.shape[0], 12, input.shape[2])
    # first 9 values are drum velocities
    for i in range(input.shape[0]):
        for j in range(input.shape[2]):
            hits = pitch_class_map[int(input[i, 0, j])]
            X_score[i, hits, j] = input[i, 1, j]
    # next 3 values are tempo, tsig, pos_in_bar
    X_score[:, 9:12, :] = input[:, 2:5, :]
    return X_score

if __name__ == '__main__':
    test = torch.tensor([[[42, 36, 38, 42, 36],
                          [70, 60, 111, 120, 101],
                          [120, 120, 140, 140, 140],
                          [1, 1, 1, 1.5, 1.5],
                          [0, 0.5, 0, 0.33, 0.66]]])
    print(readScore(test).shape)
    print(readScore(test)[:, :10, :])