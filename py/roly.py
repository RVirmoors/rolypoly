"""Rolypoly Python implementation
2020 rvirmoors

Requires pythonosc, numpy, librosa.
"""


import argparse
import queue
import sys

import pretty_midi
from constants import ROLAND_DRUM_PITCH_CLASSES

from pythonosc import udp_client
import time

import numpy as np
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt

# parse command line args
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)  # show docstring from top
parser.add_argument(
    '--drummidi', default='data/baron.mid', metavar='FOO.mid',
    help='drum MIDI file name')
parser.add_argument(
    '--firstrun', action='store_true',
    help='first run (no timing dynamics)')
parser.add_argument(
    '--offline', action='store_true',
    help='execute offline (learn)')
args = parser.parse_args()

# load MIDI file
pm = pretty_midi.PrettyMIDI(args.drummidi)

if (len(pm.instruments) > 1):
    sys.exit('There are {} instruments. Please load a MIDI file with just one\
 (drum) instrument track.'.format(len(pm.instruments)))

# vars
drumtrack = pm.instruments[0]
if (drumtrack.is_drum == False):
    sys.exit('Your MIDI file must be a DRUM track.')

client = udp_client.SimpleUDPClient("127.0.0.1", 5005)

feat_vec_size = len(ROLAND_DRUM_PITCH_CLASSES) + 3
tc = pm.get_tempo_changes()

# SETUP METHODS


def classes_to_map(classes):
    """
    Build a pitch class map. (Kicks are class 0 etc)
    """
    class_map = {}
    for cls, pitches in enumerate(classes):
        for pitch in pitches:
            class_map[pitch] = cls
    return class_map


def score_tempo():
    """
    Assign BPM values to every note
    """
    note_tempos = np.ones(len(drumtrack.notes))
    times = tc[0]
    tempos = tc[1]
    for index, note in enumerate(drumtrack.notes):
        atTime = note.start
        # get index of tempo
        here = np.searchsorted(times, atTime + 0.01)
        if here:
            here -= 1
        # the found tempo is assigned to the current note
        note_tempos[index] = tempos[here]
    return note_tempos


def score_timesig():
    """
    Assign timesig to every note
    """
    timesigs = np.full((len(drumtrack.notes), 2), 4)  # init all as [4 4]
    for index, note in enumerate(drumtrack.notes):
        atTime = note.start
        for i in pm.time_signature_changes:
            if i.time < atTime or np.isclose(i.time, atTime):
                here = i
            else:
                break
        timesigs[index] = (here.numerator, here.denominator)
    return timesigs


def score_pos_in_bar():
    positions_in_bar = np.zeros(len(drumtrack.notes))
    first_timesig = pm.time_signature_changes[0]
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

# LIVE METHODS


def processFV(featVec):
    print(featVec[11])


def parseMIDItoFV():
    """
    Play the drum MIDI file in real time (or not), emitting
    feature vectors to be processed by processFV().
    """
    start = time.monotonic()
    featVec = np.zeros(feat_vec_size)  # 9+3 zeros
    for index, note in enumerate(drumtrack.notes):
        #        print(pitch_class_map[note.pitch])
        if index < (len(drumtrack.notes) - 1):
            # if we're not at the last note, maybe wait
            currstart = note.start
            nextstart = drumtrack.notes[index + 1].start
            sleeptime = nextstart - currstart
        # one-hot encode feature vector
        featVec[pitch_class_map[note.pitch]] = 1
        if sleeptime:
            # FV complete, process it and wait for the next one
            #client.send_message("/delay", 0)
            featVec[9] = tempos[index]
            num, denom = timesigs[index][0], timesigs[index][1]
            featVec[10] = num / denom
            featVec[11] = positions_in_bar[index]
            processFV(featVec)
            featVec = np.zeros(feat_vec_size)
            if not args.offline:
                time.sleep(sleeptime)


pitch_class_map = classes_to_map(ROLAND_DRUM_PITCH_CLASSES)
tempos = score_tempo()
timesigs = score_timesig()
positions_in_bar = score_pos_in_bar()

parseMIDItoFV()
