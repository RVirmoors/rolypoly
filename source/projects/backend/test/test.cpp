#include <iostream>
#include <math.h>
#include <filesystem>
#include <torch/torch.h>
#include "backend.hpp"

using namespace torch;
using namespace at::indexing;
namespace fs = std::filesystem;

// ========== MODEL ===============

torch::Tensor toyThreshToOnes(torch::Tensor src, float thresh = 0.0) {
    return torch::where(
                src.index({Slice(), Slice(), Slice(1, 4)}) > thresh,
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
    hitsEmbedding(nn::Embedding(8, d_model)), // 2^9 possible hit combinations
    hitsTransformer(nn::TransformerEncoder(nn::TransformerEncoderOptions(nn::TransformerEncoderLayerOptions(d_model, nhead), enc_layers))),
    masker(nn::Transformer(nn::TransformerOptions())), // just to generate mask
    hitsFc(nn::Linear(d_model, 8))    
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
        torch::Tensor pos = src.index({Slice(), Slice(), Slice(0, 1)});
        
        src = toyThreshToOnes(src);
        src = backend::oneHotToInt(src, 3).to(device);// torch::cat({src, pos}, 2);

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
    float decay_ratio = (ep - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters);
    _ASSERT(0 <= decay_ratio);
    _ASSERT(decay_ratio <= 1);
    float coeff = 0.5 * (1.0 + cos(M_PI * decay_ratio));
    return config.min_lr + coeff * (config.lr - config.min_lr);
}

torch::Tensor toyHitsLoss(torch::Tensor y_hat, torch::Tensor y) {
    torch::Tensor y_hits = toyThreshToOnes(y);    
    y_hits = backend::oneHotToInt(y_hits, 3);
    y_hits = nn::functional::one_hot(y_hits, 8).to(torch::kFloat);

    // std::cout << y_hits.sizes() << " " << y_hat.sizes() << std::endl;

    return torch::cross_entropy_loss(y_hat, y_hits);
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
        static_cast<torch::optim::AdamOptions&>(optimizer.param_groups()[0].options()).lr(lr); // set lr: https://stackoverflow.com/questions/62415285/updating-learning-rate-with-libtorch-1-5-and-optimiser-options-in-c

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
    config.epochs = 2000;
    config.final = false;
    config.eval_interval = 5;
    config.eval_iters = 10; // 200
    config.lr = 6e-5;

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
    for (int i = 0; i < data.size(0) - 2; ++i) {
        torch::Tensor input = data.slice(0, i, i + 2);
        torch::Tensor output = data.slice(0, i + 1, i + 3);
        input_seq_list.push_back(input);
        output_list.push_back(output);
    }

    std::map<std::string, std::vector<torch::Tensor>> train_data;

    for (int i = 0; i < 8; i++)
        train_data["X_enc"].push_back(data);
    train_data["X_dec"] = input_seq_list;
    train_data["Y"] = output_list;

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

    std::cout << "INPUT: " << input_seq_list[1] << std::endl;
    std::cout << "TARGET: " << output_list[1] << std::endl;
    std::cout << "PREDICTION: " << model(input_seq_list[1].unsqueeze(0), input_seq_list[1].unsqueeze(0)) << std::endl;
    std::cin.get();

    std::cout << "INPUT: " << input_seq_list[3] << std::endl;
    std::cout << "TARGET: " << output_list[3] << std::endl;
    std::cout << "PREDICTION: " << model(input_seq_list[3].unsqueeze(0), input_seq_list[3].unsqueeze(0)) << std::endl;
    std::cin.get();

    return 0;
}