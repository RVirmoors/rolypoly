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


def classes_to_map(classes):
    """
    Build a pitch class map. (Kicks are class 0 etc)
    """
    class_map = {}
    for cls, pitches in enumerate(classes):
        for pitch in pitches:
            class_map[pitch] = cls
    return class_map


def processFV(featVec):
    print(featVec)


def parseMIDItoFV():
    """
    Play the drum MIDI file in real time (or not), emitting
    feature vectors to be processed by processFV().
    """
    start = time.monotonic()
    featVec = np.zeros(len(ROLAND_DRUM_PITCH_CLASSES))  # 9 zeros
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
            processFV(featVec)
            featVec = np.zeros(len(ROLAND_DRUM_PITCH_CLASSES))
            if not args.offline:
                time.sleep(sleeptime)


pitch_class_map = classes_to_map(ROLAND_DRUM_PITCH_CLASSES)
parseMIDItoFV()
