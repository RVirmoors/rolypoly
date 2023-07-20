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

const double weight_decay = 1e-1;
const double beta1 = 0.9;
const double beta2 = 0.95;

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
        std::cerr << "Error opening file: " << filename << std::endl;
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

    take = torch::zeros({ int(data.size()), int(data[0].size()) });
    for (int i = 0; i < int(data.size()); i++) {
        for (int j = 0; j < int(data[0].size()); j++) {
            take[i][j] = data[i][j];
        }
    }
}

void takeToTrainData(torch::Tensor& take, torch::Tensor& input_encode, torch::Tensor& input_decode, torch::Tensor& output_decode) {
    input_encode = take;
    auto sum_non_zero = torch::sum(input_encode.slice(1, 0, 9), 0);
    auto count_non_zero = (input_encode.narrow(1, 0, 9) != 0).sum(0);
    //std::cout << count_non_zero << std::endl;
    auto mean = sum_non_zero / count_non_zero;
    //std::cout << mean << std::endl;
    input_encode.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0,9)}, torch::where(input_encode.narrow(1, 0, 9) != 0, mean, input_encode.narrow(1, 0, 9)));
    // std::cout << input_encode << std::endl;

    input_decode = take.slice(0, 0, take.size(0) - 1);
    output_decode = take.slice(0, 1, take.size(0));
}

int main() {

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA." << std::endl;
        device = torch::kCUDA;
    }

    std::vector<MetaData> meta;
    getMeta(meta);

    backend::TrainConfig config;
    config.batch_size = 5; // 512
    config.block_size = 16;
    config.epochs = 1000;
    config.final = false;
    std::map<std::string, std::vector<torch::Tensor>> train_data, val_data;

    for (const auto& data : meta) {
        std::cout << "train? " << data.values.at("split")._Equal("train") << std::endl;
        std::string csv_filename = "groove/" + data.values.at("midi_filename").substr(0, data.values.at("midi_filename").size() - 4) + ".csv";
        torch::Tensor take;
        csvToTensor(csv_filename, take, config.block_size);
        if (take.size(0) == 0)
            break;
        take = take.to(device);
        std::cout << ": " << take.sizes() << std::endl;
        torch::Tensor xe, xd, y;
        takeToTrainData(take, xe, xd, y);

        auto split = data.values.at("split");
        if (split._Equal("train")) {
            std::cout << "add train" << std::endl;
            train_data["X_enc"].push_back(xe);
            train_data["X_dec"].push_back(xd);
            train_data["Y"].push_back(y);
        } else {
            std::cout << "add validation" << std::endl;
            val_data["X_enc"].push_back(xe);
            val_data["X_dec"].push_back(xd);
            val_data["Y"].push_back(y);
        }
    }

    backend::TransformerModel model(INPUT_DIM, OUTPUT_DIM, 128, 8, device);
    
    std::string load_model = "out/model.pt";
    if (fs::exists(load_model)) {
        try {
            torch::load(model, load_model);
            std::cout << "Model checkpoint loaded successfully from: " << load_model << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model checkpoint: " << e.what() << std::endl;
        }
    }

    backend::train(model, config, train_data, val_data, load_model, device);

    std::cin.get();
    return 0;

    // std::cout << model->forward(data[0], input_seq[0]) << std::endl;
    return 0;
}