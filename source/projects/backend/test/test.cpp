#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include "backend.hpp"

using namespace torch;
namespace fs = std::filesystem;

int main() {

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA." << std::endl;
        device = torch::kCUDA;
    }

    // backend::TransformerModel model(5, 5, 64, 8, 1, 1, device);
    backend::ToyHitsTransformer model(64, 8, 1, device);

    std::string load_model = "model.pt";
    if (fs::exists(load_model)) {
        try {
            torch::load(model, load_model);
            std::cout << "Model checkpoint loaded successfully from: " << load_model << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model checkpoint: " << e.what() << std::endl;
        }
    }

    backend::TrainConfig config;
    // TODO: make these command-line configurable
    config.batch_size = 1; // 512;
    config.block_size = 1; // 16;
    config.epochs = 1000;
    config.final = false;
    config.eval_interval = 5;
    config.eval_iters = 10; // 200
    config.lr = 6e-3;

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

    std::vector<torch::Tensor> input_seq_list, output_list;
    for (int i = 0; i < data.size(1) - 2; ++i) {
        torch::Tensor input = data.slice(0, i, i + 2);
        torch::Tensor output = data.slice(0, i + 1, i + 3);
        input_seq_list.push_back(input);
        output_list.push_back(output);
    }

    std::map<std::string, std::vector<torch::Tensor>> train_data;

    for (int i = 0; i < 3; i++)
        train_data["X_enc"].push_back(data);
    train_data["X_dec"] = input_seq_list;
    train_data["Y"] = output_list;

    std::cout << "X_enc size: " << train_data["X_enc"].size() << " x " << train_data["X_enc"][0].sizes() <<
        "\nX_dec size: " << train_data["X_dec"].size() << " x " << train_data["X_dec"][0].sizes() <<
        "\n  Y   size: " << train_data["Y"].size() << " x " << train_data["Y"][0].sizes() << std::endl;
    
    try {
        backend::train(model, config, train_data, train_data, "model.pt", device);
    } catch (const std::exception& e) {
            std::cout << e.what();
            std::cin.get();
    }

    std::cout << "INPUT: " << input_seq_list[1] << std::endl;
    std::cout << "TARGET: " << output_list[1] << std::endl;
    std::cout << "PREDICTION: " << model(input_seq_list[1].unsqueeze(0), output_list[0].unsqueeze(0)) << std::endl;
    std::cin.get();

    return 0;
}