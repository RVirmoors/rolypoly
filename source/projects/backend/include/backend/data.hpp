// Rolypoly C++ implementation
// 2023 rvirmoors
//
// Backend: data utilities

#pragma once

#include <iostream>
#include <torch/torch.h>

namespace backend {

using namespace torch;
using namespace at::indexing;

// Build a pitch class map. (Kicks are class 0 etc)
// modified from https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/data.py
const int NUM_PITCHES = 128;
const int NUM_CLASSES = 9;
std::vector<int> classes[NUM_CLASSES] = {
    // kick drum
    {35, 36},
    // snare drum
    {38, 37, 40},
    // closed hi-hat
    {42, 22, 44},
    // open hi-hat
    {46, 26},
    // low tom
    {43, 58},
    // mid tom
    {47, 45},
    // high tom
    {50, 48},
    // crash cymbal
    {49, 52, 55, 57},
    // ride cymbal
    {51, 53, 59}
};

std::array<int, NUM_PITCHES> classes_to_map() {
    std::array<int, NUM_PITCHES> class_map = {0};
    for (int cls = 0; cls < NUM_CLASSES; ++cls) {
        for (int pitch : classes[cls]) {
            class_map[pitch] = cls;
        }
    }
    return class_map;
}

void dataScaleDown(torch::Tensor& data) {
    /*
    Scale the input data to range [0, 1].

    9 velocities from [0, 127]
    9 offsets from [-0.04, 0.04]
    bpm from [40, 240]
    bar_pos keep fractional part

    input: (batch, block_size, input_dim)
    output: (batch, block_size, input_dim)
    */

    data.index({Slice(), Slice(), Slice(0, 9)}).div_(127);
    data.index({Slice(), Slice(), Slice(9, 18)}).div_(0.08);
    data.index({Slice(), Slice(), Slice(INX_BPM, INX_BPM + 1)}).sub_(40).div_(200);
    data.index({Slice(), Slice(), Slice(INX_BAR_POS, INX_BAR_POS + 1)}).frac_();
}

void dataScaleUp(torch::Tensor& data) {
    /*
    Scale back up from [0, 1]
    input: (batch, block_size, input_dim)
    output: (batch, block_size, input_dim)
    */
    data.index({Slice(), Slice(), Slice(0, 9)}).mul_(127);
    data.index({Slice(), Slice(), Slice(9, 18)}).mul_(0.08);
    data.index({Slice(), Slice(), Slice(INX_BPM, INX_BPM + 1)}).mul_(200).add_(40);
}

std::string tensor_to_csv(at::Tensor tensor) {
  // in: tensor of shape (length, channels)
  // out: csv string of shape (length, channels)
  std::string csv = "";
  for (int i = 0; i < tensor.size(0); i++) {
    for (int j = 0; j < tensor.size(1); j++) {
      csv += std::to_string(tensor[i][j].item<double>());
      if (j < tensor.size(1) - 1)
        csv += ",";
    }
    csv += "\n";
  }
  return csv;
}


} // end namespace backend