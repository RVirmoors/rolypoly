"""
Rolypoly Python implementation
2023 rvirmoors

data preprocessing methods
"""

import torch

X_ENCODER_CHANNELS = 12 # 9 drum channel velocities + bpm, tsig, pos_in_bar
X_DECODER_CHANNELS = 14 # above + tau_drum, tau_guitar
IN_DRUM_CHANNELS = 5 # hit, vel, tempo, tsig, pos_in_bar
IN_ONSET_CHANNELS = 5 # 666, tau_guitar, tempo, tsig, pos_in_bar

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

def readScore(input: torch.Tensor, m_enc_dim: int = X_ENCODER_CHANNELS):
    # input: (batch, 5, vec_size) from cpp host
    # output: (batch, 12, vec_size)
    pitch_class_map = classes_to_map()
    X_score = torch.zeros(input.shape[0], m_enc_dim, input.shape[2])
    # first 9 values are drum velocities
    for i in range(input.shape[0]):
        for j in range(input.shape[2]):
            if input[i, 0, j] != 0 and input[i, 0, j] != 666:
                hits = pitch_class_map[int(input[i, 0, j])]
                X_score[i, hits, j] = input[i, 1, j]
    # next 3 values are tempo, tsig, pos_in_bar
    X_score[:, 9:12, :] = input[:, 2:5, :]
    # remove all rows with only zeros
    mask = ~torch.all(X_score == 0, dim=1).squeeze()
    X_score = X_score[:, :, mask]
    return X_score

def readLiveOnset(input: torch.Tensor, x_dec: torch.Tensor):
    # add tau_guitar to decoder input
    # input: (batch, 5, 1) from cpp host
    # output: (batch, 14, vec_size)
    if input[:, 0, 0] != 666:
        return x_dec
    i = x_dec.shape[2] - 1
    while i >= 0:
        if input[:, 2:5, 0].all() == x_dec[:, 9:12, i].all():
            x_dec[:, 13, i] = input[:, 1, 0]
            return x_dec
        i -= 1
    return x_dec

def readScoreLive(input: torch.Tensor, x_dec: torch.Tensor):
    # input: (batch, 5, vec_size) from cpp host
    # output: (batch, 14, vec_size)
    live_notes = readScore(input) # (batch, 12, vec_size)
    live_notes = torch.cat((live_notes, torch.zeros(
        live_notes.shape[0], 2, live_notes.shape[2])), dim=1) # (batch, 14, vec_size)
    x_dec = torch.cat((x_dec, live_notes), dim=2)
    return x_dec


# === TESTS ===
if __name__ == '__main__':
    print(upbeat(0.05), upbeat(0.24))
    test = torch.tensor([[[42, 36, 38, 42, 36],
                          [70, 60, 111, 105, 101],
                          [120, 120, 140, 140, 140],
                          [1, 1, 1, 1.5, 1.5],
                          [0, 0.5, 0.33, 0.33, 0.66]]])
    print(readScore(test).shape)
    x = readScore(test)
    feat = x.squeeze(0)
    for i in range(feat.shape[1]):
        print("->", feat[:, i])
        print(bartime_to_ms(0.1, feat[:, i]))