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
        Slice(0, OUTPUT_DIM)}); // (num_samples-1, OUTPUT_DIM)
}

// ========== MODEL ===============

torch::Tensor toyThreshToOnes(torch::Tensor src, float startIndex = 9, float thresh = 0.0) {
    return torch::where(
                src.index({Slice(), Slice(), Slice(startIndex, startIndex+9)}) > thresh,
                1.0,
                0.0
            ); // replace all hits with 1
}

struct TransformerModelImpl : nn::Module {
    TransformerModelImpl(int input_dim, int output_dim, int d_model, int nhead, int enc_layers, int dec_layers, torch::Device device) :
    device(device),
    d_model(d_model),
    pos_linLayer(nn::Linear(1, d_model)),
    encEmbedding(nn::Linear(9, d_model)),
    decEmbedding(nn::Linear(input_dim-9, d_model)),
    transformer(nn::Transformer(nn::TransformerOptions(d_model, nhead, enc_layers, dec_layers))),
    fc(nn::Linear(d_model, output_dim))
    {   
        register_module("pos_linLayer", pos_linLayer);
        register_module("encEmbedding", encEmbedding);
        register_module("decEmbedding", decEmbedding);
        register_module("transformer", transformer);
        register_module("fc", fc);
        pos_linLayer->to(device);
        encEmbedding->to(device);
        decEmbedding->to(device);
        transformer->to(device);
        fc->to(device);
    }

    torch::Tensor generatePE(torch::Tensor pos) {
        torch::frac_(pos);
        return pos_linLayer(pos);
    }

    torch::Tensor forward(torch::Tensor input) {
        torch::Tensor pos = input.index({Slice(), Slice(), Slice(INX_BAR_POS, INX_BAR_POS+1)}); 
        torch::Tensor posenc = generatePE(pos);

        torch::Tensor src = input.index({Slice(), Slice(), Slice(0, 9)});
        torch::Tensor tgt = input.index({Slice(), Slice(), Slice(9, None)});

        std::cout << "SRC: " << src[0][0] << std::endl;
        std::cout << "TGT: " << tgt[0][0] << std::endl;
        std::cin.get();

        src = encEmbedding(src) + posenc;
        tgt = decEmbedding(tgt) + posenc;

        torch::Tensor src_mask = transformer->generate_square_subsequent_mask(src.size(1)).to(device);
        torch::Tensor tgt_mask = transformer->generate_square_subsequent_mask(tgt.size(1)).to(device);
            
        // (B, T, C) -> (T, B, C)
        src.transpose_(0, 1);
        tgt.transpose_(0, 1);

        torch::Tensor output = transformer(src, tgt, src_mask, tgt_mask);
        
        output.transpose_(0, 1); // (T, B, C) -> (B, T, C)
        
        output = fc(output);
        return output;
    }

    nn::Linear pos_linLayer, encEmbedding, decEmbedding, fc;
    nn::Transformer transformer;
    torch::Device device;
    int d_model;
};
TORCH_MODULE(TransformerModel);


struct ToyHitsTransformerImpl : nn::Module {
// predicting upcoming hits
    ToyHitsTransformerImpl(int d_model, int nhead, int enc_layers, torch::Device device) :
    device(device),
    d_model(d_model),
    pos_linLayer(nn::Linear(1, d_model)),
    hitsEmbedding(nn::Linear(9, d_model)), 
    hitsTransformer(nn::TransformerEncoder(nn::TransformerEncoderOptions(nn::TransformerEncoderLayerOptions(d_model, nhead), enc_layers))),
    masker(nn::Transformer(nn::TransformerOptions())), // just to generate mask
    hitsFc(nn::Linear(d_model, 1+9))    
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
        torch::frac_(pos);
        return pos_linLayer(pos);
    }

    torch::Tensor forward(torch::Tensor src) {
        torch::Tensor pos = src.index({Slice(), Slice(), Slice(INX_BAR_POS, INX_BAR_POS+1)});        
        src = toyThreshToOnes(src).to(device);

        torch::Tensor src_posenc = generatePE(pos);
        src = hitsEmbedding(src) * sqrt(d_model) + src_posenc;
        torch::Tensor src_mask = masker->generate_square_subsequent_mask(src.size(1)).to(device);
            
        src.transpose_(0, 1);    // (B, T, C) -> (T, B, C)
        torch::Tensor output = hitsTransformer(src, src_mask);
        output.transpose_(0, 1); // (T, B, C) -> (B, T, C)

        output = hitsFc(output);
        output = torch::sigmoid(output);
        return output;
    }

    nn::Linear hitsEmbedding;
    nn::TransformerEncoder hitsTransformer;
    nn::Transformer masker; // just to generate mask
    nn::Linear pos_linLayer, hitsFc;
    torch::Device device;
    int d_model;
};
TORCH_MODULE(ToyHitsTransformer);

// ========== TRAIN ===============

float get_lr(int ep, backend::TrainConfig config) {
// https://github.com/karpathy/nanoGPT/blob/master/train.py#L228C5-L228C5
    if (!config.decay_lr) {
        return config.lr;
    }
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
    torch::Tensor y_hat_hits = y_hat.index({Slice(), Slice(), Slice(1,10)});
    torch::Tensor y_hits = toyThreshToOnes(y).to(torch::kFloat);

    torch::Tensor y_hat_pos = y_hat.index({Slice(), Slice(), 0});
    torch::Tensor y_pos = torch::frac(y.index({Slice(), Slice(), INX_BAR_POS}));

    // std::cout << y_hits.sizes() << " " << y_hat.sizes() << std::endl;

    return torch::cross_entropy_loss(y_hat_hits, y_hits) + torch::mse_loss(y_hat_pos, y_pos);
}

torch::Tensor computeLoss(torch::Tensor y_hat, torch::Tensor y) {
    // discard zero note offsets from y
    auto mask = (y.index({ Slice(), Slice(), Slice(0, 9) }) == 0.0);

    y_hat.narrow(2, 9, 9).index_put_(
        { mask.to(torch::kBool) }, 
        y_hat.narrow(2, 9, 9).index({ mask.to(torch::kBool) }) * 0.0);

    return torch::mse_loss(y_hat, y);
}

void train(ToyHitsTransformer hitsModel,
            TransformerModel model,
            backend::TrainConfig config,
            std::map<std::string, std::vector<torch::Tensor>>& train_data,
            std::map<std::string, std::vector<torch::Tensor>>& val_data,
            std::string save_hits_model = "hitsModel.pt",
            std::string save_model = "model.pt",
            torch::Device device = torch::kCPU) 
{
    bool trainHits = false;
    if (hitsModel)
        trainHits = true;
    else
        hitsModel = ToyHitsTransformer(1,1,1,device); // dummy

    torch::optim::Adam hitsOptimizer(hitsModel->parameters(), torch::optim::AdamOptions(config.lr));
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(config.lr));
    double min_loss = std::numeric_limits<double>::infinity();

    std::cout << "Training Ensemble..." << std::endl;
    hitsModel->train();

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        optimizer.zero_grad();

        float lr = get_lr(epoch, config);
        for (auto param_group : hitsOptimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(lr);
        }
        for (auto param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(lr);
        } // set lr: https://stackoverflow.com/questions/62415285/updating-learning-rate-with-libtorch-1-5-and-optimiser-options-in-c

        torch::Tensor x, y;
        x = torch::stack(train_data["X"]);
        y = torch::stack(train_data["Y"]);
        backend::dataScaleDown(x);
        backend::dataScaleDown(y);
        torch::Tensor x_hits;
        torch::Tensor loss;

        if (trainHits) {
            x_hits = hitsModel->forward(x);
            loss = toyHitsLoss(x_hits, y);
            loss.backward();
            nn::utils::clip_grad_norm_(hitsModel->parameters(), 0.5);
            hitsOptimizer.step();
            // update encoder input with predicted hits
            x.index_put_({Slice(), Slice(), INX_BAR_POS},
                x_hits.index({Slice(), Slice(), 0})
                ); 
            x.index_put_({Slice(), Slice(), Slice(0, 9)},
                x_hits.index({Slice(), Slice(), Slice(1, 10)})
                );
        }

        torch::Tensor out = model->forward(x);
        loss = computeLoss(out, y);
        loss.backward();
        nn::utils::clip_grad_norm_(model->parameters(), 0.5);
        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " - train loss: " << loss.item<float>() << " | lr: " << lr << std::endl;
            if (loss.item<float>() < min_loss) {
                min_loss = loss.item<float>();
                torch::save(model, save_model);
                if (trainHits)
                    torch::save(hitsModel, save_hits_model);
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

    TransformerModel model(INPUT_DIM, OUTPUT_DIM, 128, 16, 12, 12, device);
    ToyHitsTransformer hitsModel(128, 16, 12, device);

    std::string load_model = "hitsModel.pt";
    if (fs::exists(load_model)) {
        try {
            torch::load(hitsModel, load_model);
            std::cout << "Model checkpoint loaded successfully from: " << load_model << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model checkpoint: " << e.what() << std::endl;
        }
    }
    load_model = "model.pt";
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
    config.batch_size = 32; // 512;
    config.block_size = 16; // 16;
    config.epochs = 12000;
    config.final = false;
    config.eval_interval = 5;
    config.eval_iters = 10; // 200
    config.decay_lr = true;
    config.lr = 4e-6;

    torch::Tensor take;
    csvToTensor("groovae.csv", take);
    take = take.to(device);
    std::cout << ": " << take.sizes() << std::endl;

    torch::Tensor x, y;
    takeToTrainData(take, x, y);
    
    std::map<std::string, std::vector<torch::Tensor>> train_data;

    int bs = config.block_size;
    for (int i = 0; i < x.size(0) / bs - 1; i++) {
        train_data["X"].push_back(x.index({Slice(i * bs, (i+1) * bs)}));
        train_data["Y"].push_back(y.index({Slice(i * bs, (i+1) * bs)}));
    }

    std::cout << "X size: " << train_data["X"].size() << " x " << train_data["X"][0].sizes() <<
        "\nY size: " << train_data["Y"].size() << " x " << train_data["Y"][0].sizes() << std::endl;
    
    try {
        train(nullptr, model, config, train_data, train_data, "hitsModel.pt", "model.pt", device);
    } catch (const std::exception& e) {
            std::cout << e.what();
            std::cin.get();
    }

    hitsModel->eval();

    // std::cout << "INPUT: " << train_data["X"][0][15] << std::endl;
    auto target = train_data["Y"][0][15];
    std::cout << "TARGET:     " << target[20].item<float>() << " : " << (toyThreshToOnes(target.unsqueeze(0).unsqueeze(0))) << std::endl;
    auto pred = hitsModel(train_data["X"][0].unsqueeze(0))[0][15];
    std::cout << "PREDICTION: " << pred[0].item<float>() << " : " << pred << std::endl;
    std::cin.get();

    target = train_data["Y"][1][15];
    std::cout << "TARGET:     " << target[20].item<float>() << " : " << (toyThreshToOnes(target.unsqueeze(0).unsqueeze(0))) << std::endl;
    pred = hitsModel(train_data["X"][1].unsqueeze(0))[0][15];
    std::cout << "PREDICTION: " << pred[0].item<float>() << " : " << pred << std::endl;
    std::cin.get();

    target = train_data["Y"][2][15];
    std::cout << "TARGET:     " << target[20].item<float>() << " : " << (toyThreshToOnes(target.unsqueeze(0).unsqueeze(0))) << std::endl;
    pred = hitsModel(train_data["X"][2].unsqueeze(0))[0][15];
    std::cout << "PREDICTION: " << pred[0].item<float>() << " : " << pred << std::endl;
    std::cin.get();

    return 0;
}