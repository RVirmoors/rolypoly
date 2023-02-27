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
        ms = time(s) to be converted [batch, seq_len, times_to_compute]
        featVec = feature of the note we relate to 
    """
    tempo = featVec[9]
    timeSig = featVec[10]
    barDiff = ms / 1000. * 60. / tempo / timeSig

    return barDiff

def bartime_to_ms(bartime, featVec):
    """
    Convert a bar-relative time difference to a ms interval.
    input:
        bartime = time(s) to be converted [batch, seq_len, times_to_compute]
        featVec = feature of the note we relate to
    """
    if len(bartime.shape) == 2:
        bartime = bartime.unsqueeze(2)
    tempo = featVec[:, :, 9].unsqueeze(2)
    timeSig = featVec[:, :, 10].unsqueeze(2)
    ms = bartime * 1000. / 60. * tempo * timeSig
    if ms.shape[2] == 1:
        ms = ms.view(bartime.shape[0], bartime.shape[1])
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

def saveYtoCSV(Y, filename: str) -> int:
    """
    Save Y tensor to csv file.
    input: Y (len, feat_vec_size), filename
    output: number of rows written
    """
    with open(filename, 'w') as f:
        f.write("kick, snar, hcls, hopn, ltom, mtom, htom, cras, ride, bpm, tsig, pos_in_bar, tau_d, tau_g\n")
        rows = Y.shape[0]
        for row in range(rows):
            f.write(', '.join([str(x) for x in Y[row].tolist()]) + '\n')
    return rows

def loadYFromCSV(filename: str) -> torch.Tensor:
    """
    Load Y tensor from csv file.
    input: filename
    output: Y (len, feat_vec_size)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        rows = len(lines) - 1
        cols = len(lines[1].split(', '))
        Y = torch.zeros(rows, cols)
        for row in range(rows):
            Y[row] = torch.tensor([float(x) for x in lines[row + 1].split(', ')])
    return Y

# === DATA PROCESSING ===

def readScore(input: torch.Tensor, m_enc_dim: int = X_ENCODER_CHANNELS):
    # input: (batch, score_size, 5) from cpp host
    # output: (batch, score_size, m_enc_dim = 12)
    pitch_class_map = classes_to_map()
    X_score = torch.zeros(input.shape[0], input.shape[1], m_enc_dim)
    k = 0 # write index
    # first 9 values are drum velocities
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j, 0] != 0 and input[i, j, 0] != 666:
                hits = pitch_class_map[int(input[i, j, 0])]
                X_score[i, k, hits] = input[i, j, 1]
                # next 3 values are tempo, tsig, pos_in_bar
                X_score[:, k, 9:12] = input[:, j, 2:5]
                if j < input.shape[1] - 1:
                    if input[:, j, 2] != input[:, j+1, 2] or input[:, j, 3] != input[:, j+1, 3] or input[:, j, 4] != input[:, j+1, 4]:
                        # next timestep
                        k += 1
    # remove all rows with only zeros
    mask = ~torch.all(X_score == 0, dim=2).squeeze()
    #print("mask:", mask.dim(), mask.shape, mask)
    if mask.dim() == 0:
        return X_score
    X_score = X_score[:, mask, :]
    return X_score

def readLiveOnset(input: torch.Tensor, x_dec: torch.Tensor, x_enc: torch.Tensor):
    # add tau_guitar to decoder input
    # input: (batch, 1, 5) from cpp host
    # output: (batch, vec_size, 14)
    if x_dec.shape[1] == 0:
        return x_dec
    if input[:, 0, 0] != 666:
        return x_dec
    i = x_dec.shape[1] - 1
    while i >= 0:
        print("LOOKING FOR MATCH", input[:, 0, 2:5], x_enc[:, i, 9:12])
        if torch.allclose(input[:, 0, 2:5], x_enc[:, i, 9:12]):
            x_dec[:, i, 13] = ms_to_bartime(input[:, 0, 1], x_enc[:, i].squeeze())   # tau_guitar
            print("FOUND MATCH", x_dec[:, i, 13])
            return x_dec
        i -= 1
    return x_dec

def readScoreLive(input: torch.Tensor):
    # input: (batch, vec_size, 5) from cpp host
    # output: (batch, vec_size, 14)
    live_notes = readScore(input) # (batch, vec_size, 12)
    live_notes = torch.cat((live_notes, torch.zeros(
        live_notes.shape[0], live_notes.shape[1], 2)), dim=2) # (batch, vec_size, 14)
    return live_notes

def dataScaleDown(input: torch.Tensor):
    """
    Scale the input data to range [-1, 1].

    9 velocities from [0, 127]
    bpm from [40, 240]
    tsig from [0.5, 1.5]
    pos_in_bar, tau_d, tau_g from [0, 1]

    input: (batch, vec_size, 14)
    output: (batch, vec_size, 14)
    """       
    input[:, :, :9] = input[:, :, :9] / 63.5 - 1
    input[:, :, 9] = (input[:, :, 9] - 40) / 100 - 1
    input[:, :, 10] = input[:, :, 10] - 1
    input[:, :, 11:] = input[:, :, 11:] * 2 - 1 # pos_in_bar, tau_d, tau_g are bartimes

    return input

def dataScaleUp(input: torch.Tensor):
    """
    Scale the input data back up from [-1, 1].

    9 velocities from [0, 127]
    bpm from [40, 240]
    tsig from [0.5, 1.5]
    pos_in_bar from [0, 1]

    input: (batch, vec_size, 14)
    output: (batch, vec_size, 14)
    """
    
    input[:, :, :9] = (input[:, :, :9] + 1) * 63.5
    input[:, :, 9] = (input[:, :, 9] + 1) * 100 + 40
    input[:, :, 10] = input[:, :, 10] + 1
    input[:, :, 11:] = (input[:, :, 11:] + 1) / 2

    return input


# === TESTS ===

if __name__ == '__main__':
    #print(upbeat(0.05), upbeat(0.24))
    test = torch.tensor([[  [0, 0, 0, 0, 0],
                            [42, 70, 120, 1, 0],
                            [36, 60, 120, 1, 0.5],
                            [38, 111, 140, 1.5, 0.33],
                            [42, 105, 140, 1.5, 0.33],
                            [36, 101, 140, 1.5, 0.66]]])
    test0 = torch.tensor([[[0, 0, 0, 0, 0]]])
    test1 = torch.tensor([[[36, 60, 120, 1, 0.5]]])
    x = readScore(test)
    print("readScore shape =", x.shape)
    x = torch.cat((x, torch.randn(x.shape[0], x.shape[1], 2)), dim=2)

    x_original = x.clone().detach()
    dataScaleUp(dataScaleDown(x))

    print(x_original, x)
    tolerance = 1e-3
    assert torch.allclose(x, x_original, atol=tolerance)

    feat = x.squeeze(0)
    print(feat)