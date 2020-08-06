"""Rolypoly Python implementation
2020 rvirmoors

Requires pythonosc, numpy, librosa.
"""
RUNSEQ = False

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
from torch.utils.data import DataLoader
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
    '--drummidi', default='data/baron.mid', metavar='FOO.mid',
    help='drum MIDI file name')
parser.add_argument(
    '--preload_model', default='models/last.pt', metavar='FOO.pt',
    help='start from a pre-trained model')
parser.add_argument(
    '--offline', action='store_true',
    help='execute offline (learn)')
parser.add_argument(
    '--take', default='data/takes/last.csv', metavar='FOO.csv',
    help='take csv file name for offline training')
parser.add_argument(
    '--bootstrap', action='store_true',
    help='Bootstrap LSTM with position & guitar.')
parser.add_argument(
    '--seq2seq', action='store_true',
    help='Add LSTM decoder for a Seq2Seq model.')
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
guitarDescr = 0  # kurtosis


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


async def processFV(featVec, model, X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths):
    """
    Live:
    1. send the drums to be played in Max
    2. wait a little to hear the guitar info
    3. send the drum+guitar FV to the RNN for inference
    4. wait a little to hear the guitar onset
    5. save FV & y_hat & onsetDelay for future offline training
    6. return the obtained y_hat = next beat timing
    """
    # 1.
    play = ["%.3f" % feat for feat in featVec]
    play = ' '.join(play)
    client.send_message("/play", play)
    # 2.
    await asyncio.sleep(featVec[9] * 0.1 / 1000)
    featVec[13] = guitarDescr
    # 3.
    #print(int(featVec[0]), int(featVec[1]), int(
    #    featVec[2]), int(featVec[3]), int(featVec[4]))
    if RUNSEQ:
        # if new bar, finish existing sequence and start a new one
        if featVec[12] <= X[s_i][h_i][12] and h_i:
            lastBar = X[s_i]
            newBar = np.zeros_like(lastBar)
            newBar[0] = featVec
            in_lengths = [int(X_lengths[s_i]) + 1]
        elif s_i > 0:
            lastBar = X[s_i - 1]
            newBar = X[s_i]
            newBar[h_i + 1] = featVec
            in_lengths = [int(X_lengths[s_i - 1]) + h_i + 2]
        else:
            # first bar. Second bar is still empty
            lastBar = X[s_i]
            lastBar[h_i + 1] = featVec
            newBar = np.zeros_like(lastBar)
            in_lengths = [h_i + 2]

        twoBars = np.concatenate((lastBar, newBar), axis=0)
        x = torch.Tensor(twoBars).double()  # dtype=torch.float64)
        x = x[None, :, :]       # one batch, 1 or 2 seqs
    else:
        x = torch.Tensor(featVec).double()  # dtype=torch.float64)
        x = x[None, None, :]    # one batch, one seq
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        if RUNSEQ:
            model.init_hidden()         # reset the model state
            y_hat = model(x, in_lengths)    # look at 1-2 bars
            y_hat = y_hat[-1][in_lengths[0] - 1][0]
        else:
            y_hat = model(x, [1])[0][0]     # one fV
    # 4.
    await asyncio.sleep(featVec[9] * 0.6 / 1000)
    # remains constant if no guit onset?
    onsetDelay = data.ms_to_bartime(delayms, featVec)
    print("drum-guitar: {:.4f} || next drum-delay: {:.4f}".
          format(onsetDelay, y_hat.item()))
    # 5.
    X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths = timing.addRow(
        featVec, y_hat, onsetDelay, X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths)
    # 6.
    return y_hat, X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths


async def parseMIDItoFV(model):
    """
    Play the drum MIDI file in real time (or not?), emitting
    feature vectors to be processed by processFV().
    """
    X = np.zeros((1000, 64, feat_vec_size))  # seqs * hits * features
    Y = np.zeros((1000, 64))                 # seqs * hits
    Y_hat = np.zeros((1000, 64))             # seqs * hits
    diff_hat = np.zeros((1000, 64))          # seqs * hits
    X_lengths = np.zeros(1000)
    s_i = h_i = -1
    next_delay = 0
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

            # loop = asyncio.get_event_loop()
            # next hit timing [ms]
            y_hat, X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths = await processFV(featVec, model, X, Y, Y_hat, diff_hat, h_i, s_i, X_lengths)
            prev_delay = next_delay
            next_delay = data.bartime_to_ms(y_hat.item(), featVec)

            # reset FV and wait
            featVec = np.zeros(feat_vec_size)

            client.send_message("/next", next_delay)
            await asyncio.sleep(sleeptime * 0.4 + (next_delay - prev_delay) / 1000.)
    return X, Y, Y_hat, diff_hat, s_i + 1, X_lengths


pitch_class_map = data.classes_to_map(ROLAND_DRUM_PITCH_CLASSES)
tempos = data.score_tempo(drumtrack, tc)
timesigs = data.score_timesig(drumtrack, ts)
positions_in_bar = data.score_pos_in_bar(drumtrack, ts, tempos, timesigs)


async def init_main():
    if args.offline:
        # OFFLINE : ...
        x, xl, dh, y, bs = timing.load_XY(args.take)
        x, xl, yh, dh = timing.prepare_X(x, xl, dh, dh, bs)
        batch_size = bs
        longest_seq = int(max(xl))
        y = torch.DoubleTensor(y[:batch_size, :longest_seq])
        # define model for offline: learn a batch
        model = timing.TimingLSTM(
            input_dim=feat_vec_size,
            batch_size=batch_size,
            bootstrap=args.bootstrap,
            seq2seq=args.seq2seq)

        if args.preload_model:
            trained_path = args.preload_model
            model.load_state_dict(torch.load(
                trained_path, map_location=timing.device))
            print("Loaded pre-trained model weights from", trained_path)

        train_data = [{'X': x, 'X_lengths': xl,
                       'Y': y, 'diff_hat': dh, 'split': 'train'}]
        dl = {}
        dl['train'] = DataLoader(train_data, batch_size=1,
                                 shuffle=False)

        trained_model, loss = timing.train(model, dl,
                                           minibatch_size=int(batch_size / 1),
                                           minihop_size=int(batch_size / 1),
                                           epochs=10)

        if get_y_n("Save trained model? "):
            PATH = "models/last.pt"
            torch.save(trained_model.state_dict(), PATH)
            print("Saved trained model to", PATH)
    else:
        # ONLINE :
        # listen on port 5006
        server = AsyncIOOSCUDPServer(
            ("127.0.0.1", 5006), dispatcher, asyncio.get_event_loop())
        # Create datagram endpoint and start serving
        transport, protocol = await server.create_serve_endpoint()

        # define model for LIVE.
        model = timing.TimingLSTM(
            input_dim=feat_vec_size,
            batch_size=1,
            bootstrap=args.bootstrap,
            seq2seq=args.seq2seq)

        if args.preload_model:
            trained_path = args.preload_model
            model.load_state_dict(torch.load(
                trained_path, map_location=timing.device))
            print("Loaded pre-trained model weights from", trained_path)

        client.send_message("/record", 1)
        # Enter main loop of program
        X, Y, Y_hat, diff_hat, batch_size, X_lengths = await parseMIDItoFV(model)
        client.send_message("/record", 0)

        X, X_lengths, Y_hat, diff_hat = timing.prepare_X(
            X, X_lengths, Y_hat, diff_hat, batch_size)
        Y_hat, Y = timing.prepare_Y(X_lengths, diff_hat, Y_hat, Y,
                                    style='EMA', value=0.96)

        total_loss = model.loss(Y_hat, Y, torch.DoubleTensor(diff_hat))
        print('Take loss: {:4f}'.format(total_loss))
        print('Take MSE (16th note) loss: {:4f}'.format(total_loss * 16 * 16))

        if get_y_n("Save performance? "):
            rows, filename = timing.save_XY(X, X_lengths, diff_hat, Y, Y_hat)
            print("Saved", filename, ": ", rows, "rows.")
            client.send_message("/save", filename[11:-3] + "wav")

        transport.close()  # Clean up serve endpoint


asyncio.run(init_main())


"""
TODO test: constant vs EMA
simple, boots, s2s
normal play, lazy, triplets, quantized
"""
