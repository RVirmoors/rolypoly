"""
Rolypoly Python implementation
2023 rvirmoors

data preprocessing methods
"""

import torch
import numpy as np
torch.set_printoptions(sci_mode=False, linewidth=200)

X_ENCODER_CHANNELS = 9 # 9 drum channel velocities
X_DECODER_CHANNELS = 11 # above + tau_drum, tau_guitar
X_POS_CHANNELS = 3 # bpm, tsig, bar_pos
INX_BPM = 0
INX_TSIG = 1
INX_BAR_POS = 2
IN_DRUM_CHANNELS = 5 # hit, vel, tempo, tsig, bar_pos
IN_ONSET_CHANNELS = 5 # 666, tau_guitar, tempo, tsig, bar_pos

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
    tempo = featVec[INX_BPM]
    timeSig = featVec[INX_TSIG]
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
    tempo = featVec[:, :, INX_BPM].unsqueeze(2)
    timeSig = featVec[:, :, INX_TSIG].unsqueeze(2)
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

def score_bar_pos(drumtrack, timesig_changes, tempos, timesigs):
    """
    Return np.list of positions for every note in track
    """
    bar_positions = np.zeros(len(drumtrack.notes))
    first_timesig = timesig_changes[0]
    cur_bar = 0 # current bar
    barStart = atTime = 0
    barEnd = 240 / tempos[0] * \
        first_timesig.numerator / first_timesig.denominator
    for index, note in enumerate(drumtrack.notes):
        if note.start >= atTime:
            atTime = note.start
        if np.isclose(atTime, barEnd, atol=0.1) or atTime > barEnd:
            # reached the end of the bar, compute the next one
            cur_bar += 1
            barStart = barEnd
            barEnd = barEnd + 240 / tempos[index] * \
                timesigs[index][0] / timesigs[index][1]
            # print("EOB", barStart, barEnd)
        # cur_pos is always in [0, 1)
        cur_pos = (atTime - barStart) / (barEnd - barStart)
        bar_positions[index] = cur_pos + cur_bar
        #print(atTime, ":", cur_pos + cur_bar)
    return bar_positions

# === FILE I/O ===

def saveYtoCSV(Y, Pos, filename: str) -> int:
    """
    Save Y, Pos tensors to csv file.
    input: Y (len, feat_vec_size), Pos (len, pos_size), filename
    output: number of rows written
    """
    with open(filename, 'w') as f:
        f.write("kick, snar, hcls, hopn, ltom, mtom, htom, cras, ride, tau_d, tau_g, bpm, tsig, bar_pos\n")
        rows = Y.shape[0] + Pos.shape[0]
        for row in range(rows):
            f.write(', '.join([str(x) for x in Y[row].tolist()]) + ', '.join([str(x) for x in Pos[row].tolist()]) + '\n')
    return rows

def loadYFromCSV(filename: str) -> torch.Tensor:
    """
    Load Y tensor from csv file.
    input: filename
    output: Y (len, feat_vec_size), Pos (len, pos_size)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        rows = len(lines) - 1
        cols = len(lines[1].split(', '))
        Y = torch.zeros(rows, X_DECODER_CHANNELS)
        Pos = torch.zeros(rows, X_POS_CHANNELS)
        assert cols == X_DECODER_CHANNELS + X_POS_CHANNELS
        for row in range(rows):
            Y[row] = torch.tensor([float(x) for x in lines[row + 1].split(', ')][:X_DECODER_CHANNELS])
            Pos[row] = torch.tensor([float(x) for x in lines[row + 1].split(', ')][X_DECODER_CHANNELS:])
    return Y, Pos

# === DATA PROCESSING ===

def readScore(input: torch.Tensor):
    # input: (batch, score_size, 5) from cpp host
    # output: (batch, score_size, m_enc_dim = 12), (batch, score_size, m_pos_dim = 4)
    m_enc_dim = X_ENCODER_CHANNELS
    m_pos_dim = X_POS_CHANNELS
    pitch_class_map = classes_to_map()
    X_score = torch.zeros(input.shape[0], input.shape[1], m_enc_dim)
    X_pos = torch.zeros(input.shape[0], input.shape[1], m_pos_dim)
    k = 0 # write index
    # first 9 values are drum velocities
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j, 0] != 0 and input[i, j, 0] != 666:
                hits = pitch_class_map[int(input[i, j, 0])]
                X_score[i, k, hits] = input[i, j, 1]
                # next 3 values are tempo, tsig, bar_pos
                X_pos[i, k, INX_BPM]  = input[i, j, 2]
                X_pos[i, k, INX_TSIG] = input[i, j, 3]
                X_pos[i, k, INX_BAR_POS] = input[i, j, 4] 
                if j < input.shape[1] - 1:
                    if input[:, j, 2] != input[:, j+1, 2] or input[:, j, 3] != input[:, j+1, 3] or input[:, j, 4] != input[:, j+1, 4]:
                        # next timestep
                        k += 1
    # remove any rows with only zeros
    mask = ~torch.all(X_score == 0, dim=2).squeeze()
    #print("mask:", mask.dim(), mask.shape, mask)
    if mask.dim() == 0:
        return X_score, X_pos
    X_score = X_score[:, mask, :]
    X_pos = X_pos[:, mask, :]
    return X_score, X_pos

def readLiveOnset(input: torch.Tensor, x_dec: torch.Tensor, x_pos: torch.Tensor):
    # add tau_guitar to decoder input
    # input: (batch, 1, 5) from cpp host
    # output: (batch, vec_size, 14)
    if x_dec.shape[1] == 0:
        return x_dec
    if input[:, 0, 0] != 666:
        return x_dec
    i = x_dec.shape[1] - 1
    while i >= 0: # TODO: binary search? make this more efficient
        print("LOOKING FOR MATCH", input[:, 0, 2:5], x_pos[:, i])
        if torch.allclose(input[:, 0, 2:5], x_pos[:, i):
            x_dec[:, i, 13] = ms_to_bartime(input[:, 0, 1], x_pos[:, i].squeeze())   # tau_guitar
            print("FOUND MATCH", x_dec[:, i, 13])
            return x_dec
        i -= 1
    return x_dec

def readScoreLive(input: torch.Tensor):
    # add two zeros for decoder input
    # input: (batch, vec_size, 5) from cpp host
    # output: (batch, vec_size, 11), (batch, vec_size, 4)
    live_notes, positions = readScore(input) # (batch, vec_size, 9)
    live_notes = torch.cat((live_notes, torch.zeros(
        live_notes.shape[0], live_notes.shape[1], 2)), dim=2) # (batch, vec_size, 11)
    return live_notes, positions

def dataScaleDown(input: torch.Tensor , pos: torch.Tensor):
    """
    Scale the input data to range [-1, 1].

    9 velocities from [0, 127]
    bpm from [40, 240]
    tsig from [0.5, 1.5]

    input: (batch, vec_size, 14)
    output: (batch, vec_size, 14)
    """       
    input[:, :, :9] = input[:, :, :9] / 63.5 - 1
    pos[:,:, INX_BPM] = (pos[:,:, INX_BPM] - 40) / 100 - 1
    pos[:,:, INX_TSIG] = pos[:,:, INX_TSIG] - 1

    return input, pos

def dataScaleUp(input: torch.Tensor, pos: torch.Tensor):
    """
    Scale the input data back up from [-1, 1].

    9 velocities to [0, 127]
    bpm to [40, 240]
    tsig to [0.5, 1.5]

    input: (batch, vec_size, 14)
    output: (batch, vec_size, 14)
    """
    
    input[:, :, :9] = (input[:, :, :9] + 1) * 63.5
    pos[:,:, INX_BPM] = (pos[:,:, INX_BPM] + 1) * 100 + 40
    pos[:,:, INX_TSIG] = pos[:,:, INX_TSIG] + 1

    return input, pos


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
    x, pos = readScore(test)
    print("readScore shape =", x.shape, ":", pos.shape)
    x = torch.cat((x, torch.randn(x.shape[0], x.shape[1], 2)), dim=2)

    x_original = x.clone().detach()
    x, _ = dataScaleUp(dataScaleDown(x, pos)[0], pos)

    print(x_original, "\n====\n", x)
    tolerance = 1e-100
    assert torch.allclose(x, x_original, atol=tolerance)