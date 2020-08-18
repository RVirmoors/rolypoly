"""Rolypoly Python implementation
2020 rvirmoors

Requires pythonosc, numpy, librosa.
"""
HIDDEN_DIM = 256

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
import timingMeta
import data             # data helper methods
from train_gmd import GMDdataset, pad_collate
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
    '--meta', default="data/meta/last.csv", metavar='info.csv',
    help='Meta learning dataset.')
parser.add_argument(
    '--A', default=1., type = float,
    help='A scaling value.')
parser.add_argument(
    '--B', default=1., type = float,
    help='B scaling value.')
parser.add_argument(
    '--root_dir', default='data/groove/',
    help='Root directory for validation dataset.')
parser.add_argument(
    '--valid', default=None, metavar='info.csv',
    help='Metadata file: filename of csv list of samples for validation dataset.')
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

since = time.time()


def getOnsetDiffOSC(address, *args):
    # print(f"{address}: {args[0]}")
    global delayms
    delayms = args[0]


def getGuitarDescrOSC(address, *args):
    # print(f"{address}: {args[0]}")
    global guitarDescr
    guitarDescr = args[0]


# OSC communication
client = SimpleUDPClient("127.0.0.1", 8017)  # send
dispatcher = Dispatcher()
dispatcher.map("/onset", getOnsetDiffOSC)  # receive
dispatcher.map("/descr", getGuitarDescrOSC)  # receive


async def processFV(trainer, featVec, model, X, Y_hat, h_i, s_i, X_lengths, batch_size):
    """
    Live:
       [ UPSWING   @ t - dur[t-1]/2 ]
    1. get the previous audio descriptor & diff_hat, add to featVec[t]
    2. send featVec[t] to the model for inference, get y_hat[t] offset
    3. wait dur[t-1]/2 + y_hat[t] (adjusted by y_hat[t-1] + inference time)
       [ DOWNSWING @ t ]
    4. play the current timestep hits, from FV[t]
    5. store / train on featVec[t] & y_hat[t]
    6. wait dur[t]/2 (adjusted by training time)
       [ REPEAT for all timesteps t ]
    """
    global since
    writer = trainer['writer']
    next_delay = trainer['next_delay']
    # 1.
    featVec[13] = guitarDescr
    # remains constant if no guit onset:
    # print(delayms, "clamp to", featVec[9] / -6. +
    #      next_delay, " : ", featVec[9] / 6. + next_delay)
    delayms_clamped = np.clip(delayms, featVec[9] / -6. +
                              next_delay, featVec[9] / 6. + next_delay)
    featVec[14] = data.ms_to_bartime(delayms_clamped, featVec)

    # 2.
    # print(int(featVec[0]), int(featVec[1]), int(
    #    featVec[2]), int(featVec[3]), (featVec[12]))
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.set_grad_enabled(args.train_online):
        if not args.seq2seq:
            x = torch.Tensor(featVec).double()  # dtype=torch.float64)
            x = x[None, None, :]            # one batch, one seq
            y_hat = model(x, [1])[0][0]     # one fV
        else:
            longest_seq = int(max(X_lengths))
            x = torch.Tensor(X[:batch_size, :longest_seq]).double()
            xl = X_lengths[:batch_size]
            y_s = s_i
            y_h = h_i
            if featVec[12] <= X[s_i][h_i][12] and h_i >= 0:
                y_s += 1
                y_h = 0
            else:
                y_h += 1
            y_hat = model(x, xl)[y_s][y_h]  # last FV
            # 3.
            # next hit timing [ms]
    prev_delay = next_delay
    next_delay = data.bartime_to_ms(y_hat.item(), featVec)

    client.send_message("/next", next_delay)
    trainer['next_delay'] = next_delay
    inference_time = time.time() - since
    if s_i + h_i >= 0:
        wait_time = X[s_i, h_i, 9] * 0.5 / 1000 -\
            inference_time + (next_delay - prev_delay) / 1000.
    else:
        wait_time = 0  # first note
    # print("    inference time: {:.3f}  || wait time:      {:.3f}   [sec]".
    #      format(inference_time, wait_time))
    # print("WAIT - ", wait_time)
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

    X, Y_hat, h_i, s_i, X_lengths = timing.addRow(
        featVec, y_hat, X, Y_hat, h_i, s_i, X_lengths)

    trained = False
    if args.train_online:
        # TODO: fix it!
        cols = int(max(X_lengths))
        y_hat, y = timing.prepare_Y(
            #    None, featVec[14], data.ms_to_bartime(prev_delay, featVec), style='diff', online=True)
            X_lengths[:s_i + 1], X[:s_i + 1, :cols, 14], Y_hat[:s_i + 1, :cols], style='EMA', value=0.8)
        if (featVec[9] * 0.5 / 1000 < trainer['train_time']):
            # not enough time to train: accum indices and wait
            trainer['indices'] -= 1
        else:
            model, loss = timing.trainOnline(
                model, y, y_hat, indices=trainer['indices'], epochs=1, lr=1e-3)
            w_i = sum(X_lengths)
            writer.add_scalar(
                "Loss/train", loss, w_i)
            trainer['indices'] = -1
            trained = True

    train_time = time.time() - since
    # 6.
    wait_time = featVec[9] * 0.5 / 1000 - train_time
    # print("WAIT + ", wait_time)
    await asyncio.sleep(wait_time)
    since = time.time()
    if trained:
        trainer['train_time'] = train_time
        print("    train time [s]: {:.4f} || loss:        {:.4f}".
              format(trainer['train_time'], loss))
        if (wait_time < 0):
            print("WARNING: training is causing extra delays")
    return trainer, y_hat, X, Y_hat, h_i, s_i, X_lengths


async def parseMIDItoFV(model, trainer, X, X_lengths, batch_size):
    """
    Play the drum MIDI file in real time, emitting
    feature vectors to be processed by processFV().
    """
    Y_hat = np.zeros((1000, 64))             # seqs * hits
    s_i = 0
    h_i = -1
    if args.meta:
        metaX, metaY, _ = timingMeta.load_XY(filename=args.meta)
    else:
        metaX = torch.zeros(1 + HIDDEN_DIM).double().unsqueeze(dim=0)
        metaY = torch.zeros(2).double().unsqueeze(dim=0)

    client.send_message("/record", 1)
    for i, x_len in enumerate(X_lengths[:batch_size]):
        x_len = int(x_len)
        for _, featVec in enumerate(X[i][:x_len]):
            trainer, y_hat, X, Y_hat, h_i, s_i, X_lengths = \
                await processFV(trainer, featVec, model, X, Y_hat, h_i, s_i, X_lengths, batch_size)
        diff_hat = torch.DoubleTensor(X[i, :, 14])
        varDiff = torch.var(diff_hat[Y_hat[i] != 0]).unsqueeze(
            dim=0).unsqueeze(dim=0)
        A = torch.DoubleTensor([[args.A]])
        B = torch.DoubleTensor([[args.B]])

        hid = model.hidden[0][-1]
        #print("varDiff", varDiff.size())
        #print("hidden", hid.size())
        metaX, metaY = timingMeta.add_XY(
            metaX, metaY, varDiff, hid, A, B)

    if args.meta is None:
        metaX = metaX[1:]  # remove first (zeros) row
        metaY = metaY[1:]

    timingMeta.save_XY(metaX, metaY)

    return X, Y_hat, X_lengths


def parseMIDItoX():
    """
    Load the drum MIDI file into memory (X) asynchronously,
    to be then used by the seq2seq encoder to make predictions.
    """
    X = np.zeros((1000, 64, feat_vec_size))  # seqs * hits * features
    X_lengths = np.zeros(1000)
    s_i = 0
    h_i = -1

    featVec = np.zeros(feat_vec_size)  # 9+6 zeros
    for index, note in enumerate(drumtrack.notes):
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

            X, _, h_i, s_i, X_lengths = timing.addRow(
                featVec, None, X, None, h_i, s_i, X_lengths, pre=True)

            # reset FV and go to next timestep
            featVec = np.zeros(feat_vec_size)

    print("Done preloading", s_i + 1, "bars.")

    return X, X_lengths, s_i + 1


pitch_class_map = data.classes_to_map(ROLAND_DRUM_PITCH_CLASSES)
tempos = data.score_tempo(drumtrack, tc)
timesigs = data.score_timesig(drumtrack, ts)
positions_in_bar = data.score_pos_in_bar(drumtrack, ts, tempos, timesigs)


async def init_main():
    if args.offline:
        # OFFLINE : ...
        x, xl, y, bs = timing.load_XY(args.take)
        x, xl, yh = timing.prepare_X(x, xl, None, bs)
        batch_size = bs
        longest_seq = int(max(xl))
        y = torch.DoubleTensor(y[:batch_size, :longest_seq])
        # define model for offline: learn a batch
        model = timing.TimingLSTM(
            input_dim=feat_vec_size,
            batch_size=batch_size,
            seq2seq=args.seq2seq)

        if args.preload_model:
            trained_path = args.preload_model
            model.load_state_dict(torch.load(
                trained_path, map_location=timing.device))
            print("Loaded pre-trained model weights from", trained_path)

        since = time.time()
        if args.valid:
            gmd = GMDdataset(csv_file=args.root_dir + args.valid,
                             root_dir=args.root_dir)
            val_data = [gmd[i]
                        for i in range(len(gmd)) if gmd[i]['split'] != 'dropped']

        train_data = [{'X': x, 'X_lengths': xl,
                       'Y': y, 'split': 'train'}]
        dl = {}
        dl['train'] = DataLoader(train_data, batch_size=1,
                                 shuffle=False)
        if args.valid:
            dl['val'] = DataLoader(val_data, batch_size=64,
                                   shuffle=True, num_workers=1, collate_fn=pad_collate)

        time_elapsed = time.time() - since
        print('Data loaded in {:.0f}m {:.0f}s\n==========='.format(
            time_elapsed // 60, time_elapsed % 60))

        trained_model, loss = timing.train(model, dl,
                                           minibatch_size=int(batch_size),
                                           epochs=30,
                                           lr=1e-3)

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

        X, X_lengths, batch_size = parseMIDItoX()

        # define model for LIVE.
        model = timing.TimingLSTM(
            input_dim=feat_vec_size,
            batch_size=batch_size if args.seq2seq else 1,
            seq2seq=args.seq2seq)

        if args.preload_model:
            trained_path = args.preload_model
            model.load_state_dict(torch.load(
                trained_path, map_location=timing.device))
            print("Loaded pre-trained model weights from", trained_path)

        trainer = {}
        trainer['next_delay'] = 0
        trainer['train_time'] = 0.15    # seconds
        trainer['indices'] = -1
        trainer['writer'] = SummaryWriter()
        # Enter main loop of program
        X, Y_hat, X_lengths = await parseMIDItoFV(model, trainer, X, X_lengths, batch_size)
        client.send_message("/record", 0)

        X, X_lengths, yh = timing.prepare_X(
            X, X_lengths, Y_hat, batch_size)
        Y_hat, Y = timing.prepare_Y(X_lengths, X[:, :, 14], yh, A=args.A, B=args.B,
                                    # style='diff') # JUST FOR TESTING
                                    # style='EMA', value=0.8)
                                    style='constant')

        total_loss = model.loss(Y_hat, Y, X[:, :, 14])
        print('Take loss: {:4f}'.format(total_loss))
        print('Take MSE (16th note) loss: {:4f}'.format(total_loss * 16 * 16))

        if get_y_n("Save performance? "):
            rows, filename = timing.save_XY(X, X_lengths, Y, Y_hat)
            client.send_message("/save", filename[11:-3] + "wav")
            Y_hat, Y = timing.prepare_Y(X_lengths, X[:, :, 14], yh, A=args.A*1.1, B=args.B*1.1)
            _,_ = timing.save_XY(X, X_lengths, Y, Y_hat, filename = filename[:-4]+'_AABB.csv')
            Y_hat, Y = timing.prepare_Y(X_lengths, X[:, :, 14], yh, A=args.A*1.1, B=args.B*0.909)
            _,_ = timing.save_XY(X, X_lengths, Y, Y_hat, filename = filename[:-4]+'_AAb.csv')
            Y_hat, Y = timing.prepare_Y(X_lengths, X[:, :, 14], yh, A=args.A*0.909, B=args.B*1.1)
            _,_ = timing.save_XY(X, X_lengths, Y, Y_hat, filename = filename[:-4]+'_aBB.csv')
            Y_hat, Y = timing.prepare_Y(X_lengths, X[:, :, 14], yh, A=args.A*0.909, B=args.B*0.909)
            _,_ = timing.save_XY(X, X_lengths, Y, Y_hat, filename = filename[:-4]+'_ab.csv')


        if args.train_online and get_y_n("Save trained model? "):
            PATH = "models/last.pt"
            torch.save(model.state_dict(), PATH)
            print("Saved trained model to", PATH)
            writer.add_hparams({'layers': model.nb_layers, 'lstm_units': model.nb_lstm_units, 'lr': lr, 'epochs': epochs},
                               {'hparam/best_val_loss': best_loss, 'hparam/test_loss': total_loss})

            writer.flush()

        transport.close()  # Clean up serve endpoint

if __name__ == '__main__':
    asyncio.run(init_main())


"""
TODO test: constant vs EMA vs no-guit
simple, boots, s2s
normal play, lazy, triplets, quantized
"""
