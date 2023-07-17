"""
Rolypoly Python implementation
2023 rvirmoors

data preprocessing methods
"""

import torch
import numpy as np
torch.set_printoptions(sci_mode=False, linewidth=200)

import constants



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
    tempo = featVec[constants.INX_BPM]
    timeSig = featVec[constants.INX_TSIG]
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
    tempo = featVec[:, :, constants.INX_BPM].unsqueeze(2)
    timeSig = featVec[:, :, constants.INX_TSIG].unsqueeze(2)
    ms = bartime * 1000. / 60. * tempo * timeSig
    if ms.shape[2] == 1:
        ms = ms.view(bartime.shape[0], bartime.shape[1])
    return ms

def offbeat(bartime: torch.Tensor) -> bool:
    """
    Check if a bar-relative time is on an offbeat.
    input:
        bartime = time to be checked
    """
    bartime = torch.tensor(bartime)
    for i in range(5):
        if torch.isclose(bartime, torch.tensor(i/4), atol=0.05):
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

def saveTakeToCSV(X_take, filename: str) -> int:
    """
    Save X_take tensor to csv file.
    input: X_take (len, feat_vec_size), filename
    output: number of rows written
    """
    with open(filename, 'w') as f:
        f.write("kick, snar, hcls, hopn, ltom, mtom, htom, cras, ride, bpm, tsig, bar_pos, tau_d, tau_g\n")
        rows = X_take.shape[0]
        for row in range(rows):
            f.write(', '.join([str(x) for x in X_take[row].tolist()]) + '\n')
    return rows

def loadX_takeFromCSV(filename: str) -> torch.Tensor:
    """
    Load X_take tensor from csv file.
    input: filename
    output: X_take (len, feat_vec_size)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        rows = len(lines) - 1
        cols = len(lines[1].split(', '))
        X_take = torch.zeros(rows, constants.X_DECODER_CHANNELS)
        assert cols == constants.X_DECODER_CHANNELS
        for row in range(rows):
            X_take[row] = torch.tensor([float(x) for x in lines[row + 1].split(', ')])
    return X_take

# === DATA PROCESSING ===

def readScore(input: torch.Tensor):
    # input: (batch, score_size, 5) from cpp host
    # output: (batch, score_size, m_enc_dim = 12)
    m_enc_dim = constants.X_ENCODER_CHANNELS
    pitch_class_map = classes_to_map()
    X_score = torch.zeros(input.shape[0], input.shape[1], m_enc_dim)
    k = 0 # write index
    # first 9 values are drum velocities
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j, 0] != 0 and input[i, j, 0] != 666:
                hits = pitch_class_map[int(input[i, j, 0])]
                X_score[i, k, hits] = input[i, j, 1]
                # next 3 values are tempo, tsig, bar_pos
                X_score[i, k, constants.INX_BPM]  = input[i, j, 2]
                X_score[i, k, constants.INX_TSIG] = input[i, j, 3]
                X_score[i, k, constants.INX_BAR_POS] = input[i, j, 4] 
                if j < input.shape[1] - 1:
                    if input[:, j, 2] != input[:, j+1, 2] or input[:, j, 3] != input[:, j+1, 3] or input[:, j, 4] != input[:, j+1, 4]:
                        # next timestep
                        k += 1
    # remove any rows with only zeros
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
    while i >= 0: # TODO: binary search? make this more efficient
        # print("LOOKING FOR MATCH", input[:, 0, 2:5], x_enc[:, i, constants.INX_BPM:])
        if torch.allclose(input[:, 0, 2:5], x_enc[:, i, constants.INX_BPM:], atol=0.01):
            tau_g = ms_to_bartime(input[0, 0, 1].item(), x_enc[:, i].squeeze()) # tau_guitar
            x_dec[:, i, constants.INX_TAU_G] = tau_g  
            # print("FOUND MATCH", x_dec[:, i, constants.INX_TAU_G])
            return x_dec
        i -= 1
    return x_dec

def readScoreLive(input: torch.Tensor):
    # add two zeros for decoder input
    # input: (batch, vec_size, 5) from cpp host
    # output: (batch, vec_size, 14), (batch, vec_size, 4)
    live_notes = readScore(input) # (batch, vec_size, 12)
    live_notes = torch.cat((live_notes, torch.zeros(
        live_notes.shape[0], live_notes.shape[1], 2)), dim=2) # (batch, vec_size, 14)
    return live_notes

def dataScaleDown(input: torch.Tensor):
    """
    Scale the input data to range [0, 1].

    9 velocities from [0, 127]
    bpm from [40, 240]
    tsig from [0.5, 1.5]

    input: (batch, vec_size, 14)
    output: (batch, vec_size, 14)
    """       
    input[:, :, :9] = input[:, :, :9] / 127
    input[:,:, constants.INX_BPM] = (input[:,:, constants.INX_BPM] - 40) / 200
    input[:,:, constants.INX_TSIG] = input[:,:, constants.INX_TSIG] - 0.5

    return input

def dataScaleUp(input: torch.Tensor):
    """
    Scale the input data back up from [0, 1].

    9 velocities to [0, 127]
    bpm to [40, 240]
    tsig to [0.5, 1.5]

    input: (batch, vec_size, 14)
    output: (batch, vec_size, 14)
    """
    
    input[:, :, :9] = input[:, :, :9] * 127
    input[:,:, constants.INX_BPM] = input[:,:, constants.INX_BPM] * 200 + 40
    input[:,:, constants.INX_TSIG] = input[:,:, constants.INX_TSIG] + 0.5

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
    # append random tau_d, tau_g
    x = torch.cat((x, torch.randn(x.shape[0], x.shape[1], 2)), dim=2)

    x_original = x.clone().detach()
    x = dataScaleUp(dataScaleDown(x))

    print(x_original, "\n====\n", x)
    tolerance = 1e-100
    assert torch.allclose(x, x_original, atol=tolerance)