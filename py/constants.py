X_ENCODER_CHANNELS = 12 # 9 drum channel velocities, bpm, tsig, bar_pos
X_DECODER_CHANNELS = 14 # above + tau_drum, tau_guitar
INX_BPM = 9
INX_TSIG = 10
INX_BAR_POS = 11
INX_TAU_D = 12
INX_TAU_G = 13
IN_DRUM_CHANNELS = 5 # hit, vel, bpm, tsig, bar_pos
IN_ONSET_CHANNELS = 5 # 666, tau_guitar, bpm, tsig, bar_pos

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
lr = 3e-4 # learning rate for finetuning
epochs = 20 # epochs for finetuning

gmd_tau_d_avg = 0 # TODO: calculate these from the dataset
gmd_tau_d_std = 0.01