"""Rolypoly Python implementation
2020 rvirmoors

Requires pythonosc, numpy, librosa.
"""
TRAIN_ONLINE = True

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
# see https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
from torch.utils.tensorboard import SummaryWriter

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
    '--seq2seq', action='store_true',
    help='Add LSTM decoder for a Seq2Seq model.')
parser.add_argument(
    '--bootstrap', action='store_true',
    help='Bootstrap Seq2Seq with position & guitar.')
parser.add_argument(
    '--train_online', action='store_true',
    help='Online training of model during performance.')
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


async def processFV(trainer, featVec, model, X, Y_hat, h_i, s_i, X_lengths):
    """
    Live:
    UPSWING
    1. get the previous guitar info & g_d onset delay
    2. send the drum+guitar FV to the RNN for inference
    3. wait dur/2 + microtiming (adjusted by prev + inference time)
    DOWNSWING
    4. send the drums to be played in Max
    5. save + use FV & y_hat for online training
    6. return the full FV + y_hat
       wait dur/2 (adjusted by training time)
    """
    since = time.time()
    writer = trainer['writer']
    next_delay = trainer['next_delay']
    # 1.
    featVec[13] = guitarDescr
    # remains constant if no guit onset:
    featVec[14] = data.ms_to_bartime(delayms, featVec)

    # 2.
    # print(int(featVec[0]), int(featVec[1]), int(
    #    featVec[2]), int(featVec[3]), int(featVec[9]))
    x = torch.Tensor(featVec).double()  # dtype=torch.float64)
    x = x[None, None, :]    # one batch, one seq
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.set_grad_enabled(args.train_online):
        y_hat = model(x, [1])[0][0]     # one fV

    # 3.
    # next hit timing [ms]
    prev_delay = next_delay
    next_delay = data.bartime_to_ms(y_hat.item(), featVec)

    client.send_message("/next", next_delay)
    inference_time = time.time() - since
    wait_time = featVec[9] * 0.5 / 1000 - \
        inference_time + (next_delay - prev_delay) / 1000.
    trainer['next_delay'] = next_delay
    print("    inference time: {:.3f}  || wait time:      {:.3f}   [sec]".
          format(inference_time, wait_time))
    if (wait_time < 0):
        print("WARNING: inference is causing extra delays")
    await asyncio.sleep(wait_time)

    # 4.
    # DOWNSWING
    since = time.time()
    play = ["%.3f" % feat for feat in featVec]
    play = ' '.join(play)
    client.send_message("/play", play)

    # 5.
    print("    drum-guitar:    {:.4f} || next microtime: {:.4f}  [/bar]".
          format(featVec[14], y_hat.item()))

    if args.train_online:
        _, y = timing.prepare_Y(
            #    None, featVec[14], data.ms_to_bartime(prev_delay, featVec), style='diff', online=True)
            X_lengths, X[:, :, 14], Y_hat, style='EMA', value=0.8)
        if (featVec[9] * 0.5 / 1000 < trainer['train_time']):
            # not enough time to train: accum indices and wait
            trainer['indices'] -= 1
            trained = False
        else:
            model, loss = timing.trainOnline(
                model, y, y_hat, indices=trainer['indices'], epochs=1, lr=1e-3)
            w_i = sum(X_lengths)
            writer.add_scalar(
                "Loss/train", loss, w_i)
            trainer['indices'] = -1
            trained = True

    X, Y_hat, h_i, s_i, X_lengths = timing.addRow(
        featVec, y_hat, X, Y_hat, h_i, s_i, X_lengths)

    train_time = time.time() - since
    if trained:
        trainer['train_time'] = train_time

    # 6.
    wait_time = featVec[9] * 0.5 / 1000 - train_time
    if trained:
        print("    train time [s]: {:.4f} || loss:        {:.4f}".
              format(trainer['train_time'], loss))
        if (wait_time < 0):
            print("WARNING: training is causing extra delays")
    await asyncio.sleep(wait_time)
    return trainer, y_hat, X, Y_hat, h_i, s_i, X_lengths


async def parseMIDItoFV(model, trainer):
    """
    Play the drum MIDI file in real time, emitting
    feature vectors to be processed by processFV().
    """
    X = np.zeros((1000, 64, feat_vec_size))  # seqs * hits * features
    Y_hat = np.zeros((1000, 64))             # seqs * hits
    X_lengths = np.zeros(1000)
    s_i = 0
    h_i = -1

    start = time.monotonic()
    featVec = np.zeros(feat_vec_size)  # 9+6 zeros
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

            trainer, y_hat, X, Y_hat, h_i, s_i, X_lengths = await processFV(trainer, featVec, model, X, Y_hat, h_i, s_i, X_lengths)
            # reset FV and wait
            featVec = np.zeros(feat_vec_size)

    return X, Y_hat, s_i + 1, X_lengths


pitch_class_map = data.classes_to_map(ROLAND_DRUM_PITCH_CLASSES)
tempos = data.score_tempo(drumtrack, tc)
timesigs = data.score_timesig(drumtrack, ts)
positions_in_bar = data.score_pos_in_bar(drumtrack, ts, tempos, timesigs)


async def init_main():
    if args.offline:
        # OFFLINE : ...
        x, xl, dh, y, bs = timing.load_XY(args.take)
        x, xl, yh = timing.prepare_X(x, xl, yh, bs)
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
                       'Y': y, 'split': 'train'}]
        dl = {}
        dl['train'] = DataLoader(train_data, batch_size=1,
                                 shuffle=False)

        trained_model, loss = timing.train(model, dl,
                                           minibatch_size=int(batch_size / 2),
                                           epochs=20)

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
        trainer = {}
        trainer['next_delay'] = 0
        trainer['train_time'] = 0.15    # seconds
        trainer['indices'] = -1
        trainer['writer'] = SummaryWriter()
        # Enter main loop of program
        X, Y_hat, batch_size, X_lengths = await parseMIDItoFV(model, trainer)
        client.send_message("/record", 0)

        X, X_lengths, Y_hat = timing.prepare_X(
            X, X_lengths, Y_hat, batch_size)
        Y_hat, Y = timing.prepare_Y(X_lengths, X[:, :, 14], Y_hat,
                                    style='constant', value = 0) # JUST FOR TESTING
                                    # style='EMA', value=0.8)

        total_loss = model.loss(Y_hat, Y, None)
        print('Take loss: {:4f}'.format(total_loss))
        print('Take MSE (16th note) loss: {:4f}'.format(total_loss * 16 * 16))

        if get_y_n("Save performance? "):
            rows, filename = timing.save_XY(X, X_lengths, Y, Y_hat)
            print("Saved", filename, ": ", rows, "rows.")
            client.send_message("/save", filename[11:-3] + "wav")

        if args.train_online and get_y_n("Save trained model? "):
            PATH = "models/last.pt"
            torch.save(model.state_dict(), PATH)
            print("Saved trained model to", PATH)
            writer.add_hparams({'layers': model.nb_layers, 'lstm_units': model.nb_lstm_units, 'lr': lr, 'epochs': epochs},
                               {'hparam/best_val_loss': best_loss, 'hparam/test_loss': total_loss})

            writer.flush()

        transport.close()  # Clean up serve endpoint


asyncio.run(init_main())


"""
TODO test: constant vs EMA vs no-guit
simple, boots, s2s
normal play, lazy, triplets, quantized
"""
