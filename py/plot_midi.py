import pretty_midi
import numpy as np
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt


def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))


pm = pretty_midi.PrettyMIDI('data/baron.mid')


plt.figure(figsize=(12, 4))
plot_piano_roll(pm, 24, 44)
plt.show()

print('There are {} time signature changes'.format(
    len(pm.time_signature_changes)))
print('There are {} instruments'.format(len(pm.instruments)))
print('Instrument 3 has {} notes'.format(len(pm.instruments[0].notes)))
print('Instrument 4 has {} pitch bends'.format(
    len(pm.instruments[0].pitch_bends)))
print('Instrument 5 has {} control changes'.format(
    len(pm.instruments[0].control_changes)))
print(pm.instruments[0].notes)
