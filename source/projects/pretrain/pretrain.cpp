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
namespace fs = std::filesystem;

struct MetaData {
    std::map<std::string, std::string> values;
};

void getMeta(std::vector<MetaData>& meta) {
    std::vector<std::string> headerKeys; // Vector to store the header keys

    std::ifstream file("groove/miniinfo.csv");
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }

    std::string line;
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

    if (data.size() < minRows + 1) {
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

void takeToTrainData(torch::Tensor& take, torch::Tensor& input_encode, torch::Tensor& input_decode, torch::Tensor& output_decode) {
    input_encode = take; // (num_samples, INPUT_DIM)

    // replace velocities with mean vel of that hit
    // TODO: decide if this is necessary
    // auto sum_non_zero = torch::sum(input_encode.slice(1, 0, 9), 0);
    // auto count_non_zero = (input_encode.narrow(1, 0, 9) != 0).sum(0);
    // auto mean = sum_non_zero / count_non_zero;
    // input_encode.index_put_({ Slice(), Slice(0,9)}, 
    //     torch::where(input_encode.narrow(1, 0, 9) != 0, mean, input_encode.narrow(1, 0, 9)));

    input_decode = take.slice(0, 0, take.size(0) - 1);                      // (num_samples-1, INPUT_DIM)
    output_decode = take.slice(0, 1, take.size(0)).slice(1, 0, OUTPUT_DIM); // (num_samples-1, OUTPUT_DIM)
}

int main() {

    std::string outDir = "out";
    std::string load_model = "out/model.pt";
    if (!fs::exists(outDir))
        fs::create_directory(outDir);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA." << std::endl;
        device = torch::kCUDA;
    }

    std::vector<MetaData> meta;
    getMeta(meta);

    backend::TrainConfig config;
    // TODO: make these command-line configurable
    config.batch_size = 512; // 512;
    config.block_size = 16; // 16;
    config.epochs = 10000;
    config.final = false;
    config.eval_interval = 25;
    config.eval_iters = 50; // 200
    config.lr = 6e-4;
    std::map<std::string, std::vector<torch::Tensor>> train_data, val_data;

    for (const auto& data : meta) {
        //std::cout << "train? " << data.values.at("split")._Equal("train") << std::endl;
        std::string csv_filename = "groove/" + data.values.at("midi_filename").substr(0, data.values.at("midi_filename").size() - 4) + ".csv";
        torch::Tensor take;
        csvToTensor(csv_filename, take, config.block_size);
        if (take.size(0)) {
            take = take.to(device);
            std::cout << ": " << take.sizes() << std::endl;
            torch::Tensor xe, xd, y;
            takeToTrainData(take, xe, xd, y);

            auto split = data.values.at("split");
            if (split._Equal("train")) {
                //std::cout << "add train" << std::endl;
                train_data["X_enc"].push_back(xe);
                train_data["X_dec"].push_back(xd);
                train_data["Y"].push_back(y);
            } else {
                //std::cout << "add validation" << std::endl;
                val_data["X_enc"].push_back(xe);
                val_data["X_dec"].push_back(xd);
                val_data["Y"].push_back(y);
            }
        }
    }

    #define HITGEN
    #ifdef HITGEN          
        backend::HitsTransformer model(256, 32, 6, device);
    #else
        backend::TransformerModel model(INPUT_DIM, OUTPUT_DIM, 256, 32, 6, 6, device);
    #endif



    if (fs::exists(load_model)) {
        try {
            torch::load(model, load_model);
            std::cout << "Model checkpoint loaded successfully from: " << load_model << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model checkpoint: " << e.what() << std::endl;
        }
    }
    std::cout << model << std::endl;

    backend::train(model, config, train_data, val_data, load_model, device);

    std::cout << "EXAMPLE EVAL:\n=============\n  y_hat     y" << std::endl;
    model->eval();
    torch::Tensor xe = val_data["X_enc"][0].slice(0, 0, config.block_size).unsqueeze(0);
    torch::Tensor xd = val_data["X_dec"][0].slice(0, 0, config.block_size).unsqueeze(0);
    backend::dataScaleDown(xe);
    backend::dataScaleDown(xd);
    torch::Tensor y = val_data["Y"][0].slice(0, 0, config.block_size).unsqueeze(0);
    torch::Tensor y_hat = model->forward(xe, xd);
    backend::dataScaleDown(y);
    #ifdef HITGEN 
        y = y.slice(2, 0, 12);
    #endif

    std::cout << torch::stack({y_hat[0][config.block_size-1], y[0][config.block_size-1]}, 1 )  << std::endl;
    std::cin.get();
    return 0;
}