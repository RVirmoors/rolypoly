// Rolypoly C++ implementation
// 2023 rvirmoors
//
// Backend: misc utilities

#pragma once

#include <iostream>
#include <torch/torch.h>

namespace utils {

using namespace torch;
using namespace at::indexing;

unsigned power_ceil(unsigned x) {
  if (x <= 1)
    return 1;
  int power = 2;
  x--;
  while (x >>= 1)
    power <<= 1;
  return power;
}

c74::min::path get_latest_model(std::string model_path) {
  if (model_path.substr(model_path.length() - 3) != ".pt")
    model_path = model_path + ".pt";
  return path(model_path);
}

} // end namespace backend