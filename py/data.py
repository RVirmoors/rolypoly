"""Rolypoly Python implementation
2020 rvirmoors

Data helper methods
"""

import numpy as np


def classes_to_map(classes):
    """
    Build a pitch class map. (Kicks are class 0 etc)
    """
    class_map = {}
    for cls, pitches in enumerate(classes):
        for pitch in pitches:
            class_map[pitch] = cls
    return class_map


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
    barStart = 0
    barEnd = 240 / tempos[0] * \
        first_timesig.numerator / first_timesig.denominator
    for index, note in enumerate(drumtrack.notes):
        atTime = note.start
        if np.isclose(atTime, barEnd) or atTime > barEnd:
            # reached the end of the bar, compute the next one
            barStart = atTime
            barEnd = atTime + 240 / tempos[index] * \
                timesigs[index][0] / timesigs[index][1]
            cur_pos = 0
        else:
            # pos in bar is always in [0, 1)
            cur_pos = (atTime - barStart) / (barEnd - barStart)
        positions_in_bar[index] = cur_pos
    return positions_in_bar


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
