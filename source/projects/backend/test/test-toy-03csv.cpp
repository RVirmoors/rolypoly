#include <iostream>
#include <math.h>
#include <filesystem>
#include <torch/torch.h>
#include "backend.hpp"

using namespace torch;
using namespace at::indexing;
namespace fs = std::filesystem;

// ========== DATA ================

void csvToTensor(const std::string& filename, torch::Tensor& take) {
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

    take = torch::zeros({ int(data.size()), INPUT_DIM });
    for (int i = 0; i < int(data.size()); i++) {
        for (int j = 0; j < int(data[0].size()); j++) {
            take[i][j] = data[i][j];
        }
    }
}

void takeToTrainData(torch::Tensor& take, torch::Tensor& input_encode, torch::Tensor& input_decode, torch::Tensor& output_decode) {
    input_encode = take; // (num_samples, INPUT_DIM)
    input_decode = take.slice(0, 0, take.size(0) - 1); // (num_samples-1, INPUT_DIM)
    output_decode = take.slice(0, 1, take.size(0)).slice(1, 0, OUTPUT_DIM); // (num_samples-1, OUTPUT_DIM)
}

// ========== MODEL ===============

torch::Tensor toyThreshToOnes(torch::Tensor src, float thresh = 0.0) {
    return torch::where(
                src.index({Slice(), Slice(), Slice(0, 9)}) > thresh,
                1.0,
                0.0
            ); // replace all hits with 1
}

struct ToyHitsTransformerImpl : nn::Module {
// predicting upcoming hits
    ToyHitsTransformerImpl(int d_model, int nhead, int enc_layers, torch::Device device) :
    device(device),
    d_model(d_model),
    pos_linLayer(nn::Linear(1, d_model)),
    hitsEmbedding(nn::Embedding(512, d_model)), // 2^9 possible hit combinations
    hitsTransformer(nn::TransformerEncoder(nn::TransformerEncoderOptions(nn::TransformerEncoderLayerOptions(d_model, nhead), enc_layers))),
    masker(nn::Transformer(nn::TransformerOptions())), // just to generate mask
    hitsFc(nn::Linear(d_model, 1+512))    
    {
        register_module("pos_linLayer", pos_linLayer);
        register_module("hitsEmbedding", hitsEmbedding);
        register_module("hitsTransformer", hitsTransformer);
        register_module("masker", masker);
        register_module("hitsFc", hitsFc);

        pos_linLayer->to(device);
        hitsEmbedding->to(device);
        hitsTransformer->to(device);
        masker->to(device);
        hitsFc->to(device);
    }

    torch::Tensor generatePE(torch::Tensor pos) {
        return pos_linLayer(pos);
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt /*not used*/) {
        torch::Tensor pos = src.index({Slice(), Slice(), Slice(20, 21)});        
        src = toyThreshToOnes(src);
        src = backend::oneHotToInt(src).to(device);// torch::cat({src, pos}, 2);

        torch::Tensor src_posenc = generatePE(pos);
        src = hitsEmbedding(src) * sqrt(d_model) + src_posenc;
        torch::Tensor src_mask = masker->generate_square_subsequent_mask(src.size(1)).to(device);
            
        src.transpose_(0, 1);    // (B, T, C) -> (T, B, C)
        torch::Tensor output = hitsTransformer(src, src_mask);
        output.transpose_(0, 1); // (T, B, C) -> (B, T, C)

        output = hitsFc(output);
        //output = torch::argmax(output, 2);
        //output = intToOneHot(output);
        return output;
    }

    nn::Embedding hitsEmbedding;
    nn::TransformerEncoder hitsTransformer;
    nn::Transformer masker; // just to generate mask
    nn::Linear pos_linLayer, hitsFc;
    torch::Device device;
    double d_model;
};
TORCH_MODULE(ToyHitsTransformer);

// ========== TRAIN ===============

float get_lr(int ep, backend::TrainConfig config) {
// https://github.com/karpathy/nanoGPT/blob/master/train.py#L228C5-L228C5
    if (ep < config.warmup_iters) {
        return config.lr * ep / config.warmup_iters;
    }
    if (ep > config.lr_decay_iters) {
        return config.min_lr;
    }
    float decay_ratio = (float)(ep - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters);
    _ASSERT(0 <= decay_ratio);
    _ASSERT(decay_ratio <= 1);
    float coeff = 0.5 * (1.0 + cos(M_PI * decay_ratio));
    return config.min_lr + coeff * (config.lr - config.min_lr);
}

torch::Tensor toyHitsLoss(torch::Tensor y_hat, torch::Tensor y) {
    torch::Tensor y_hat_hits = y_hat.index({Slice(), Slice(), Slice(1,513)});
    torch::Tensor y_hits = toyThreshToOnes(y);
    y_hits = backend::oneHotToInt(y_hits);
    y_hits = nn::functional::one_hot(y_hits, 512).to(torch::kFloat);

    torch::Tensor y_hat_pos = y_hat.index({Slice(), Slice(), 0});
    torch::Tensor y_pos = y.index({Slice(), Slice(), INX_BAR_POS});

    // std::cout << y_hits.sizes() << " " << y_hat.sizes() << std::endl;

    return torch::cross_entropy_loss(y_hat_hits, y_hits) + torch::mse_loss(y_hat_pos, y_pos);
}

void train(ToyHitsTransformer model,
            backend::TrainConfig config,
            std::map<std::string, std::vector<torch::Tensor>>& train_data,
            std::map<std::string, std::vector<torch::Tensor>>& val_data,
            std::string save_model = "hit_model.pt",
            torch::Device device = torch::kCPU) 
{
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(config.lr));
    //torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));
    double min_loss = std::numeric_limits<double>::infinity();

    std::cout << "Training Hits Generator..." << std::endl;
    model->train();

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        optimizer.zero_grad();

        float lr = get_lr(epoch, config);
        for (auto param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(lr);
        } // set lr: https://stackoverflow.com/questions/62415285/updating-learning-rate-with-libtorch-1-5-and-optimiser-options-in-c

        torch::Tensor x_enc, x_dec, y;
        x_enc = torch::stack(train_data["X_enc"]);
        x_dec = torch::stack(train_data["X_dec"]);
        y = torch::stack(train_data["Y"]);

        //std::cout << x_enc.sizes() << " " << x_dec.sizes() << " " << y.sizes() << std::endl;

        torch::Tensor y_hat = model->forward(x_dec, x_dec);
        torch::Tensor loss = toyHitsLoss(y_hat, y);
        loss.backward();
        nn::utils::clip_grad_norm_(model->parameters(), 0.5);
        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " - train loss: " << loss.item<float>() << " | lr: " << lr << std::endl;
            if (loss.item<float>() < min_loss) {
                min_loss = loss.item<float>();
                torch::save(model, save_model);
            }
        }
    }
}

// ========== MAIN ===============

int main() {

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA." << std::endl;
        device = torch::kCUDA;
    }

    // backend::TransformerModel model(5, 5, 64, 8, 1, 1, device);
    ToyHitsTransformer model(128, 16, 12, device);

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
    config.batch_size = 5; // 512;
    config.block_size = 16; // 16;
    config.epochs = 12000;
    config.final = false;
    config.eval_interval = 5;
    config.eval_iters = 10; // 200

    torch::Tensor take;
    csvToTensor("groovae.csv", take);
    take = take.to(device);
    std::cout << ": " << take.sizes() << std::endl;

    torch::Tensor xe, xd, y;
    takeToTrainData(take, xe, xd, y);
    
    std::map<std::string, std::vector<torch::Tensor>> train_data;

    int bs = config.block_size;
    for (int i = 0; i < xd.size(0) / bs - 1; i++) {
        train_data["X_enc"].push_back(xe);
        train_data["X_dec"].push_back(xd.index({Slice(i * bs, (i+1) * bs)}));
        train_data["Y"].push_back(y.index({Slice(i * bs, (i+1) * bs)}));
    }

    std::cout << "X_enc size: " << train_data["X_enc"].size() << " x " << train_data["X_enc"][0].sizes() <<
        "\nX_dec size: " << train_data["X_dec"].size() << " x " << train_data["X_dec"][0].sizes() <<
        "\n  Y   size: " << train_data["Y"].size() << " x " << train_data["Y"][0].sizes() << std::endl;
    
    try {
        train(model, config, train_data, train_data, "model.pt", device);
    } catch (const std::exception& e) {
            std::cout << e.what();
            std::cin.get();
    }

    model->eval();

    // std::cout << "INPUT: " << train_data["X_dec"][0][15] << std::endl;
    auto target = train_data["Y"][0][15];
    std::cout << "TARGET:     " << target[20].item<float>() << " : " << backend::oneHotToInt(toyThreshToOnes(target.unsqueeze(0).unsqueeze(0))).item<int>() << std::endl;
    auto pred = model(train_data["X_dec"][0].unsqueeze(0), train_data["X_dec"][0].unsqueeze(0))[0][15];
    std::cout << "PREDICTION: " << pred[0].item<float>() << " : " << torch::argmax(pred).item<int>() << std::endl;
    std::cin.get();

    return 0;
}