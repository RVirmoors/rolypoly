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
import data             # data helper methods
from constants import ROLAND_DRUM_PITCH_CLASSES
from helper import get_y_n

np.set_printoptions(suppress=True)

# parse command line args
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)  # show docstring from top
parser.add_argument(
    '--drummidi', default='data/baron3bar.mid', metavar='FOO.mid',
    help='drum MIDI file name')
parser.add_argument(
    '--take', default='data/takes/20200714123034.csv', metavar='FOO.csv',
    help='take csv file name')
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
ts = pm.time_signature_changes

delayms = 1
guitarDescr = 0  # loudness delta?

# define model for LIVE. TODO init weights
model = timing.TimingLSTM(input_dim=feat_vec_size, batch_size=1)


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
    5. save FV & y_hat & onsetDelay for future offline training
    6. return the obtained y_hat = next beat timing
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
    #print(featVec[9], featVec[10], featVec[11], featVec[12], featVec[13])
    input = torch.Tensor(featVec)
    input = input[None, None, :]    # one batch, one seq
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        y_hat = model(input, [1])           # one fV
    # 4.
    await asyncio.sleep(featVec[9] * 0.4 / 1000)
    # remains constant if no guit onset?
    onsetDelay = data.ms_to_bartime(delayms, featVec)
    print("onset delay: ", onsetDelay)
    # 5.
    timing.addRow(featVec, y_hat, onsetDelay)
    # 6.
    return y_hat


async def parseMIDItoFV():
    """
    Play the drum MIDI file in real time (or not?), emitting
    feature vectors to be processed by processFV().
    """
    start = time.monotonic()
    featVec = np.zeros(feat_vec_size)  # 9+4+1 zeros
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

            await asyncio.sleep(sleeptime + y_hat / 1000.)


pitch_class_map = data.classes_to_map(ROLAND_DRUM_PITCH_CLASSES)
tempos = data.score_tempo(drumtrack, tc)
timesigs = data.score_timesig(drumtrack, ts)
positions_in_bar = data.score_pos_in_bar(drumtrack, ts, tempos, timesigs)


async def init_main():
    if args.offline:
        # OFFLINE : ...
        # redefine model: TODO copy weights from existing model
        timing.load_XY(args.take)
        timing.prepare_X()
        batch_size = timing.s_i + 1
        longest_seq = int(max(timing.X_lengths))
        timing.Y = torch.Tensor(timing.Y[:batch_size, :longest_seq])
        model = timing.TimingLSTM(
            input_dim=feat_vec_size, batch_size=timing.s_i + 1)
        timing.train(model, batch_size)

    else:
        # ONLINE :
        # listen on port 5006
        server = AsyncIOOSCUDPServer(
            ("127.0.0.1", 5006), dispatcher, asyncio.get_event_loop())
        # Create datagram endpoint and start serving
        transport, protocol = await server.create_serve_endpoint()

        await parseMIDItoFV()  # Enter main loop of program

        timing.prepare_X()
        timing.prepare_Y()

        if get_y_n("Save performance? "):
            timing.save_XY()

        transport.close()  # Clean up serve endpoint


asyncio.run(init_main())
