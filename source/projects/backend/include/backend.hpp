#pragma once

#define ENCODER_DIM 10 // bar_pos, 9 drum hits
#define OUTPUT_DIM 18 // 9 drum channel velocities+offsets
#define TARGET_DIM 21 // 9 drum channel velocities+offsets, bpm, tsig, bar_pos
#define INPUT_DIM 22 // 9 drum channel velocities+offsets, bpm, tsig, bar_pos, tau_guitar
#define INX_BPM 18
#define INX_TSIG 19
#define INX_BAR_POS 20
#define INX_TAU_G 21
#define IN_DRUM_CHANNELS 5 // hit, vel, bpm, tsig, bar_pos
#define IN_ONSET_CHANNELS 5 // 666, tau_guitar, bpm, tsig, bar_pos

#include "backend/model.hpp"
#include "backend/train.hpp"
#include "backend/data.hpp"
#include "backend/utils.hpp"