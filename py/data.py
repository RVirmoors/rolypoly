"""
Rolypoly Python implementation
2023 rvirmoors

data preprocessing methods
"""

import torch
import numpy as np

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

# === PRETTY MIDI SCORE PARSING ===

def score_tempo(drumtrack, tempo_changes):
    """
    Assign BPM values to every note.
    """
    note_tempos = np.ones(len(drumtrack.notes))
    times = tempo_changes[0]
    tempos = tempo_changes[1]
    for index, note in enumerate(drumtrack.notes):
        atTime = note.start
        # get index of tempo
        here = np.searchsorted(times, atTime + 0.01)
        if here:
            here -= 1
        # the found tempo is assigned to the current note
        note_tempos[index] = tempos[here]
    return note_tempos

def score_timesig(drumtrack, timesig_changes):
    """
    Assign timesig to every note.
    """
    timesigs = np.full((len(drumtrack.notes), 2), 4)  # init all as [4 4]
    for index, note in enumerate(drumtrack.notes):
        atTime = note.start
        for i in timesig_changes:
            if i.time < atTime or np.isclose(i.time, atTime):
                here = i
            else:
                break
        timesigs[index] = (here.numerator, here.denominator)
    return timesigs

def score_pos_in_bar(drumtrack, timesig_changes, tempos, timesigs):
    """
    Return np.list of positions in bar for every note in track
    """
    positions_in_bar = np.zeros(len(drumtrack.notes))
    first_timesig = timesig_changes[0]
    barStart = atTime = 0
    barEnd = 240 / tempos[0] * \
        first_timesig.numerator / first_timesig.denominator
    for index, note in enumerate(drumtrack.notes):
        if note.start >= atTime:
            atTime = note.start
        if np.isclose(atTime, barEnd) or atTime > barEnd:
            # reached the end of the bar, compute the next one
            barStart = barEnd
            barEnd = barEnd + 240 / tempos[index] * \
                timesigs[index][0] / timesigs[index][1]
            # print("EOB", barStart, barEnd)
        # pos in bar is always in [0, 1)
        cur_pos = (atTime - barStart) / (barEnd - barStart)
        positions_in_bar[index] = cur_pos
        #print(atTime, cur_pos)
    return positions_in_bar

# === FILE I/O ===

def saveXdecToCSV(X_dec, filename: str) -> int:
    """
    Save X_dec tensor to csv file.
    input: X_dec (len, feat_vec_size), filename
    output: number of rows written
    """
    with open(filename, 'w') as f:
        f.write("kick, snar, hcls, hopn, ltom, mtom, htom, cras, ride, bpm, tsig, pos_in_bar, tau_d, tau_g\n")
        rows = X_dec.shape[0]
        for row in range(rows):
            f.write(', '.join([str(x) for x in X_dec[row].tolist()]) + '\n')
    return rows

def loadXdecFromCSV(filename: str) -> torch.Tensor:
    """
    Load X_dec tensor from csv file.
    input: filename
    output: X_dec (len, feat_vec_size)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        rows = len(lines) - 1
        cols = len(lines[1].split(', '))
        X_dec = torch.zeros(rows, cols)
        for row in range(rows):
            X_dec[row] = torch.tensor([float(x) for x in lines[row + 1].split(', ')])
    return X_dec

# === DATA PROCESSING ===

def readScore(input: torch.Tensor, m_enc_dim: int = X_ENCODER_CHANNELS):
    # input: (batch, 5, vec_size) from cpp host
    # output: (batch, 12, vec_size)
    pitch_class_map = classes_to_map()
    X_score = torch.zeros(input.shape[0], m_enc_dim, input.shape[2])
    k = 0 # write index
    # first 9 values are drum velocities
    for i in range(input.shape[0]):
        for j in range(input.shape[2]):
            if input[i, 0, j] != 0 and input[i, 0, j] != 666:
                hits = pitch_class_map[int(input[i, 0, j])]
                X_score[i, hits, k] = input[i, 1, j]
                # next 3 values are tempo, tsig, pos_in_bar
                X_score[:, 9:12, k] = input[:, 2:5, j]
                if j < input.shape[2] - 1:
                    if input[:, 2, j] != input[:, 2, j + 1] or input[:, 3, j] != input[:, 3, j + 1] or input[:, 4, j] != input[:, 4, j + 1]:
                        # next timestep
                        k += 1
    # remove all rows with only zeros
    mask = ~torch.all(X_score == 0, dim=1).squeeze()
    print("mask:", mask.dim(), mask.shape, mask)
    if mask.dim() == 0:
        return X_score
    X_score = X_score[:, :, mask]
    return X_score

def readLiveOnset(input: torch.Tensor, x_dec: torch.Tensor):
    # add tau_guitar to decoder input
    # input: (batch, 5, 1) from cpp host
    # output: (batch, 14, vec_size)
    if x_dec.shape[2] == 0:
        return x_dec
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
    #print(upbeat(0.05), upbeat(0.24))
    test = torch.tensor([[[0, 42, 36, 38, 42, 36],
                          [0, 70, 60, 111, 105, 101],
                          [0, 120, 120, 140, 140, 140],
                          [0, 1, 1, 1.5, 1.5, 1.5],
                          [0, 0, 0.5, 0.33, 0.33, 0.66]]])
    test0 = torch.tensor([[[0],[0],[0],[0],[0]]])
    test1 = torch.tensor([[[38],[95],[150],[1],[0.25]]])
    x = readScore(test)
    print("readScore shape =", x.shape)
    feat = x.squeeze(0)
    print(feat)