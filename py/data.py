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

def ms_to_bartime(ms: float, featVec):
    """
    Convert a ms time difference to a bar-relative diff.
    input:
        ms = time to be converted
        featVec = feature of the note we relate to
    """
    tempo = featVec[9]
    timeSig = featVec[10]
    barDiff = ms / 1000 * 60 / tempo / timeSig
    return barDiff


def bartime_to_ms(bartime: float, featVec):
    """
    Convert a bar-relative time difference to a ms interval.
    input:
        bartime = time to be converted
        featVec = feature of the note we relate to
    """
    tempo = featVec[9]
    timeSig = featVec[10]
    ms = bartime * 1000 / 60 * tempo * timeSig
    return ms

def upbeat(bartime: torch.Tensor) -> bool:
    """
    Check if a bar-relative time is on an upbeat.
    input:
        bartime = time to be checked
    """
    bartime = torch.tensor(bartime)
    if torch.isclose(bartime, torch.tensor(0.), atol=0.05):
        return False
    if torch.isclose(bartime, torch.tensor(0.25), atol=0.05):
        return False
    if torch.isclose(bartime, torch.tensor(0.5), atol=0.05):
        return False
    if torch.isclose(bartime, torch.tensor(0.75), atol=0.05):
        return False
    if torch.isclose(bartime, torch.tensor(1.), atol=0.05):
        return False
    return True

# === DATA PROCESSING ===

def readScore(input: torch.Tensor):
    # input: (batch, 5, vec_size) from cpp host
    # output: (batch, 13, vec_size)
    pitch_class_map = classes_to_map()
    X_score = torch.zeros(input.shape[0], 13, input.shape[2])
    # first 9 values are drum velocities
    for i in range(input.shape[0]):
        for j in range(input.shape[2]):
            hits = pitch_class_map[int(input[i, 0, j])]
            X_score[i, hits, j] = input[i, 1, j]
    # next 3 values are tempo, tsig, pos_in_bar
    X_score[:, 9:12, :] = input[:, 2:5, :]
    return X_score



# === TESTS ===
if __name__ == '__main__':
    print(upbeat(0.05), upbeat(0.24))
    test = torch.tensor([[[42, 36, 38, 42, 36],
                          [70, 60, 111, 105, 101],
                          [120, 120, 140, 140, 140],
                          [1, 1, 1, 1.5, 1.5],
                          [0, 0.5, 0.33, 0.33, 0.66]]])
    print(readScore(test).shape)
    #print(readScore(test)[:, :10, :])
    x = readScore(test)
    feat = x.squeeze(0)
    for i in range(feat.shape[1]):
        print("->", feat[:, i])
        print(bartime_to_ms(0.1, feat[:, i]))