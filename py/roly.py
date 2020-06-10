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

# load MIDI file
pm = pretty_midi.PrettyMIDI('data/baron.mid')

if (len(pm.instruments) > 1):
    sys.exit('There are {} instruments. Please load a MIDI file with just one\
 (drum) instrument track.'.format(len(pm.instruments)))

# vars
drumtrack = pm.instruments[0]
if (drumtrack.is_drum == False):
    sys.exit('Your MIDI file must be a DRUM track.')

client = udp_client.SimpleUDPClient("127.0.0.1", 5005)

# drum map


def classes_to_map(classes):
    class_map = {}
    for cls, pitches in enumerate(classes):
        for pitch in pitches:
            class_map[pitch] = cls
    return class_map


pitch_class_map = classes_to_map(ROLAND_DRUM_PITCH_CLASSES)

# parse drumtrack
start = time.monotonic()
for index, note in enumerate(drumtrack.notes):
    print(pitch_class_map[note.pitch])
    if index < (len(drumtrack.notes) - 1):
        # if we're not at the last note, maybe wait
        currstart = note.start
        nextstart = drumtrack.notes[index + 1].start
        sleeptime = nextstart - currstart
    if (sleeptime):
        client.send_message("/delay", 0)
        time.sleep(sleeptime)
