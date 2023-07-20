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

#define OUTPUT_DIM 21 // 9 drum channel velocities+offsets, bpm, tsig, bar_pos
#define INPUT_DIM 22 // above + tau_guitar
#define INX_BPM 18
#define INX_TSIG 19
#define INX_BAR_POS 20
#define INX_TAU_G 21
#define IN_DRUM_CHANNELS 5 // hit, vel, bpm, tsig, bar_pos
#define IN_ONSET_CHANNELS 5 // 666, tau_guitar, bpm, tsig, bar_pos

const double weight_decay = 1e-1;
const double beta1 = 0.9;
const double beta2 = 0.95;

using namespace torch;
using namespace backend;
namespace fs = std::filesystem;

struct MetaData {
    std::map<std::string, std::string> values;
};

void getMeta(std::vector<MetaData>& meta) {
    std::vector<std::string> headerKeys; // Vector to store the header keys

    std::ifstream file("groove/info.csv");
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

void csvToTensor(const std::string& filename, torch::Tensor& take) {
    std::ifstream file(filename);
    std::vector<std::vector<float>> data;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    } else {
        std::cout << "Opened file: " << filename << std::endl;
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

    std::map<std::string, std::vector<torch::Tensor>> train_data, val_data;

    for (const auto& data : meta) {
        std::cout << "train? " << data.values.at("split")._Equal("train") << std::endl;
        std::string csv_filename = "groove/" + data.values.at("midi_filename").substr(0, data.values.at("midi_filename").size() - 4) + ".csv";
        std::cout << csv_filename << " - ";
        torch::Tensor take;
        csvToTensor(csv_filename, take);
        take = take.to(device);
        std::cout << take.sizes() << std::endl;
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

    std::cin.get();
    return 0;


    TransformerModel model(INPUT_DIM, OUTPUT_DIM, 128, 8, device);
    
    std::string load_model = "out/model.pt";
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

    std::cout << model->forward(data[0], input_seq[0]) << std::endl;
    return 0;
}