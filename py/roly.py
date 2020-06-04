import pretty_midi
import numpy as np
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt

pm = pretty_midi.PrettyMIDI('data/baron.mid')

if (len(pm.instruments) == 1):
    print('There are {} instruments. Please load a MIDI file with just one\
 (drum) instrument track.'.format(len(pm.instruments)))
print('There are {} time signature changes'.format(
    len(pm.time_signature_changes)))
print(pm.instruments[0].notes)
