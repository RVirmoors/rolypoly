#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include "backend.hpp"

using namespace torch;
using namespace backend;
namespace fs = std::filesystem;

int main() {

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA." << std::endl;
        device = torch::kCUDA;
    }

    TransformerModel model(5, 5, 64, 8, device);
    
    std::string load_model = "model.pt";
    if (fs::exists(load_model)) {
        try {
            torch::load(model, load_model);
            std::cout << "Model checkpoint loaded successfully from: " << load_model << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model checkpoint: " << e.what() << std::endl;
        }
    }

    torch::Tensor data = torch::tensor({
        {0., 0.8, 0., 0.8, 0.},
        {0.5, 0., 1., 0.9, 0.007},
        {0., 0.6, 0., 0.8, 0.002},
        {0.25, 0., 0., 0.4, -0.01},
        {0.5, 0., 1., 0.7, 0.002},
        {0.75, 0., 0., 0.45, -0.005},
        {0., 0.7, 0., 0.9, 0.001},
        {0.25, 0.6, 0., 0.8, -0.002},
        {0.5, 0.2, 0.9, 0.8, 0.005},
        {0.75, 0.5, 0., 0.6, 0.002}
    }).to(device);
    // std::cout << data << std::endl;

    data = torch::stack({data, data, data, data, data, data, data, data}); // 8 batches -> ENCODER

    std::vector<torch::Tensor> input_seq_list, output_list;
    for (int i = 0; i < data.size(1) - 2; ++i) {
        torch::Tensor input = torch::stack({data[i].slice(0, i, i + 2)});
        torch::Tensor output = torch::stack({data[i].slice(0, i + 1, i + 3)});
        input_seq_list.push_back(input);
        output_list.push_back(output);
    }

    // Stack the list of tensors to create input_seq and output tensors.
    torch::Tensor input_seq = torch::stack(input_seq_list);
    torch::Tensor output = torch::stack(output_list);

    input_seq = input_seq.squeeze(1);
    output = output.squeeze(1);

    // std::cout << data.sizes() << " " << input_seq.sizes() << " " << output.sizes() << std::endl;
    // std::cout << "input_seq:" << std::endl << input_seq << std::endl;
    // std::cout << "output:" << std::endl << output << std::endl;
    // std::cout << "data:" << std::endl << data << std::endl;

    train(model, data, input_seq, output, "model.pt", device);
    return 0;
}