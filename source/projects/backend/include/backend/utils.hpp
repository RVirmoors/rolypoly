// Rolypoly C++ implementation
// 2023 rvirmoors
//
// Backend: misc utilities

#pragma once

#include <iostream>
#include "c74_min.h"
#include <torch/torch.h>

namespace utils {

using namespace torch;
using namespace at::indexing;

c74::min::path get_latest_model(std::string model_path) {
  if (model_path.substr(model_path.length() - 3) != ".pt")
    model_path = model_path + ".pt";
  return path(model_path);
}

void fill_with_zero(audio_bundle output) {
  for (int c(0); c < output.channel_count(); c++) {
    auto out = output.samples(c);
    for (int i(0); i < output.frame_count(); i++) {
      out[i] = 0.;
    }
  }
}

} // end namespace backend