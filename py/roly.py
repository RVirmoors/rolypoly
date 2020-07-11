"""Rolypoly Python implementation
2020 rvirmoors

Requires pythonosc, numpy, librosa.
"""


import argparse
import queue
import sys

import pretty_midi

from pythonosc.udp_client import SimpleUDPClient
import time

from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio

import numpy as np
import torch
import timing           # ML timing module
from constants import ROLAND_DRUM_PITCH_CLASSES

# parse command line args
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)  # show docstring from top
parser.add_argument(
    '--drummidi', default='data/baron3bar.mid', metavar='FOO.mid',
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

feat_vec_size = timing.feat_vec_size
tc = pm.get_tempo_changes()

delayms = 1
guitarDescr = 0  # loudness delta?

# define model for LIVE. TODO init weights
model = timing.TimingLSTM(input_dim=feat_vec_size, batch_size=1)

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


def getOnsetDiffOSC(address, *args):
    # print(f"{address}: {args[0]}")
    global delayms
    delayms = args[0]


def getGuitarDescrOSC(address, *args):
    # print(f"{address}: {args[0]}")
    global guitarDescr
    guitarDescr = args[0]


# OSC communication
client = SimpleUDPClient("127.0.0.1", 5005)  # send
dispatcher = Dispatcher()
dispatcher.map("/onset", getOnsetDiffOSC)  # receive
dispatcher.map("/descr", getGuitarDescrOSC)  # receive


async def processFV(featVec):
    """
    Live:
    1. send the drums to be played in Max
    2. wait a little to hear the guitar info
    3. send the drum+guitar FV to the RNN for inference
    4. wait a little to hear the guitar onset
    5. save FV & Y & onsetDelay for future offline training
    6. return the obtained Y = next beat timing
    """
    global model
    # 1.
    play = ["%.3f" % feat for feat in featVec]
    play = ' '.join(play)
    client.send_message("/play", play)
    # 2.
    await asyncio.sleep(featVec[9] * 0.1 / 1000)
    featVec[13] = guitarDescr
    # 3.
    print(featVec[9], featVec[10], featVec[11], featVec[12], featVec[13])
    input = torch.Tensor(featVec)
    input = input[None, None, :]    # one batch, one seq
    y = model(input, [1])           # one fV
    # 4.
    await asyncio.sleep(featVec[9] * 0.4 / 1000)
    # remains constant if no guit onset?
    onsetDelay = ms_to_bartime(delayms, featVec)
    print("onset delay: ", onsetDelay)
    # 5.
    timing.addRow(featVec, y, onsetDelay)
    # 6.
    return y


async def parseMIDItoFV():
    """
    Play the drum MIDI file in real time (or not?), emitting
    feature vectors to be processed by processFV().
    """
    start = time.monotonic()
    featVec = np.zeros(feat_vec_size)  # 9+4 zeros
    for index, note in enumerate(drumtrack.notes):
        #        print(pitch_class_map[note.pitch])
        if index < (len(drumtrack.notes) - 1):
            # if we're not at the last note, maybe wait
            currstart = note.start
            nextstart = drumtrack.notes[index + 1].start
            sleeptime = nextstart - currstart
        # one-hot encode feature vector [0...8]
        featVec[pitch_class_map[note.pitch]] = 1
        if sleeptime:
            # FV complete, process it and wait for the next one
            featVec[9] = sleeptime * 1000.  # hit duration [ms]
            featVec[10] = tempos[index]
            # num / denom (e.g. 4/4 = 1.)
            featVec[11] = timesigs[index][0] / timesigs[index][1]
            featVec[12] = positions_in_bar[index]

            #loop = asyncio.get_event_loop()
            y_hat = await processFV(featVec)  # next hit timing [ms]
            #y = loop.run_until_complete(infer)
            featVec = np.zeros(feat_vec_size)
            if not args.offline:
                # time.sleep(sleeptime + y / 1000.)
                await asyncio.sleep(sleeptime + y_hat / 1000.)


pitch_class_map = classes_to_map(ROLAND_DRUM_PITCH_CLASSES)
tempos = score_tempo()
timesigs = score_timesig()
positions_in_bar = score_pos_in_bar()


async def init_main():
    # listen on port 5006
    server = AsyncIOOSCUDPServer(
        ("127.0.0.1", 5006), dispatcher, asyncio.get_event_loop())
    # Create datagram endpoint and start serving
    transport, protocol = await server.create_serve_endpoint()

    await parseMIDItoFV()  # Enter main loop of program

    # OFFLINE
    timing.prepare_X()
    timing.prepare_Y()
    # redefine model: TODO copy weights from existing model
    model = timing.TimingLSTM(input_dim=feat_vec_size, batch_size=timing.s_i)

    transport.close()  # Clean up serve endpoint


asyncio.run(init_main())
