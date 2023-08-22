// Rolypoly C++ implementation
// 2023 rvirmoors
//
// Train network using Groove MIDI Dataset (GMD) from Magenta:
// https://magenta.tensorflow.org/datasets/groove


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <filesystem>
#include <torch/torch.h>
#include "backend.hpp"

using namespace torch;
using namespace at::indexing;
namespace fs = std::filesystem;

struct MetaData {
    std::map<std::string, std::string> values;
};

void getMeta(std::vector<MetaData>& meta) {
    std::ifstream file("groove/miniinfo.csv");
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }

    std::string line;
    std::vector<std::string> headerKeys; 
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string column;
        while (std::getline(iss, column, ',')) {
            headerKeys.push_back(column);
        }
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        MetaData metaData;
        std::string column;
        size_t columnIndex = 0;
        while (std::getline(iss, column, ',')) {
            metaData.values[headerKeys[columnIndex]] = column;
            columnIndex++;
        }
        meta.push_back(metaData);
    }
}

void csvToTensor(const std::string& filename, torch::Tensor& take, int minRows = 16) {
    std::ifstream file(filename);
    std::vector<std::vector<float>> data;

    if (!file.is_open()) {
        std::cerr << "No file (too short?): " << filename << std::endl;
        return;
    } else {
        std::cout << "Opened file: " << filename;
    }

    std::string line;
    std::getline(file, line); // skip the header line
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        data.push_back(row);
    }

    file.close();

    if (data.size() < minRows + 2) {
        std::cout << ", too short, discarding." << std::endl;
        return;
    }

    take = torch::zeros({ int(data.size()), INPUT_DIM });
    for (int i = 0; i < int(data.size()); i++) {
        for (int j = 0; j < int(data[0].size()); j++) {
            take[i][j] = data[i][j];
        }
    }
}

void takeToTrainData(torch::Tensor& take, torch::Tensor& input, torch::Tensor& output) {
    torch::Tensor input_enc = take.index({
        Slice(1, take.size(0)), 
        Slice(0, 9)});
    torch::Tensor input_dec = take.index({
        Slice(0, take.size(0)-1), 
        Slice(9, None)});
    input = torch::cat({input_enc, input_dec}, 1); // (num_samples-1, INPUT_DIM)
    output = take.index({
        Slice(1, take.size(0)),
        Slice(0, TARGET_DIM)}); // (num_samples-1, TARGET_DIM)
}

int main() {
    backend::TrainConfig config;
    // TODO: make these command-line configurable
    config.batch_size = 512; // 512;
    config.block_size = 16; // 16;
    config.epochs = 12000;
    config.final = false;
    config.eval_interval = 25;
    config.eval_iters = 50; // 200
    config.decay_lr = true;
    config.lr = 4e-5;
    config.train_ensemble = false;
    config.train_main = false;
    const bool dont_train = true;

    std::string outDir = "out";
    if (!fs::exists(outDir))
        fs::create_directory(outDir);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA." << std::endl;
        device = torch::kCUDA;
    }

    std::vector<MetaData> meta;
    getMeta(meta);

    std::map<std::string, std::vector<torch::Tensor>> train_data, val_data;

    for (const auto& data : meta) {
        std::string csv_filename = "groove/" + data.values.at("midi_filename").substr(0, data.values.at("midi_filename").size() - 4) + ".csv";
        torch::Tensor take;
        csvToTensor(csv_filename, take, config.block_size);
        if (take.size(0)) {
            take = take.to(device);
            std::cout << ": " << take.sizes() << std::endl;
            torch::Tensor x, y;
            takeToTrainData(take, x, y);

            auto split = data.values.at("split");
            if (split._Equal("train")) {
                //std::cout << "add train" << std::endl;
                train_data["X"].push_back(x);
                train_data["Y"].push_back(y);
            } else {
                //std::cout << "add validation" << std::endl;
                val_data["X"].push_back(x);
                val_data["Y"].push_back(y);
            }
        }
    }

    backend::TransformerModel model(INPUT_DIM, OUTPUT_DIM, 128, 16, 12, 12, device);
    backend::HitsTransformer hitsModel(128, 16, 12, device);

    std::string load_hits_model = "out/hitsModel.pt";
    if (fs::exists(load_hits_model)) {
        try {
            torch::load(hitsModel, load_hits_model, device);
            std::cout << "Model checkpoint loaded successfully from: " << load_hits_model << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model checkpoint: " << e.what() << std::endl;
        }
    }
    std::string load_model = "out/model.pt";
    if (fs::exists(load_model)) {
        try {
            torch::load(model, load_model, device);
            std::cout << "Model checkpoint loaded successfully from: " << load_model << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model checkpoint: " << e.what() << std::endl;
        }
    }

    if (!dont_train) {
        try {
        if (config.train_ensemble)
            backend::train(hitsModel, model, config, train_data, val_data, load_hits_model, load_model, device);
        else if (config.train_main)
            backend::train(nullptr, model, config, train_data, val_data, load_hits_model, load_model, device);
        else
            backend::train(hitsModel, nullptr, config, train_data, val_data, load_hits_model, load_model, device);
        } 
        catch (const std::exception& e) {
            std::cout << e.what();
            std::cin.get();
        }
    }


    try {
    std::cout << "EXAMPLE EVAL:\n=============\n  hits   offsets    y" << std::endl;
    torch::NoGradGuard no_grad_guard;
    hitsModel->train();
    model->train();
    torch::Tensor x = val_data["X"][0].slice(0, 0, config.block_size).unsqueeze(0);
    backend::dataScaleDown(x);
    torch::Tensor y = val_data["Y"][0].slice(0, 0, config.block_size).unsqueeze(0);
    backend::dataScaleDown(y);
    torch::Tensor hits = hitsModel(x);
    torch::Tensor hit_pos = hits.index({Slice(), Slice(), 0});
    hits = torch::cat({hits.index({Slice(), Slice(), Slice(1, 10)}), 
        torch::zeros({hits.size(0), hits.size(1), 9}).to(device)}, 2);

    torch::Tensor pred = model(x);

    torch::Tensor x_pos = x.index({Slice(), Slice(), INX_BAR_POS});
    torch::Tensor y_pos = y.index({Slice(), Slice(), INX_BAR_POS});
    y = y.index({Slice(), Slice(), Slice(0, 18)});
    pred = pred.index({Slice(), Slice(), Slice(0, 18)});

    std::cout << torch::stack({hits[0][config.block_size-1], pred[0][config.block_size-1], y[0][config.block_size-1]}, 1 ) << std::endl;

    std::cout << "BAR POS EVAL:\n============\n   x      hits      y\n";
    std::cout << torch::stack({x_pos[0], hit_pos[0], y_pos[0]}, 1 ) << std::endl;

    std::cout << x[0][2] << std::endl;
    // std::cout << hitsModel->parameters()[5][0] << std::endl;

    std::cin.get();
    } catch (const std::exception& e) {
            std::cout << e.what();
            std::cin.get();
    }
    return 0;
}