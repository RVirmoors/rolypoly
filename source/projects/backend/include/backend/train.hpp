// Rolypoly C++ implementation
// 2023 rvirmoors
//
// Backend: training the network

#pragma once

#include <iostream>
#include <torch/torch.h>

namespace backend {

using namespace torch;
using namespace at::indexing;

struct TrainConfig {
    int batch_size; // Batch size: how many minibatches to process at a time
    int block_size; // Block / minibatch size: how many notes to look at.
    int epochs;     // How many epochs to train for.
    int eval_interval; // How often to evaluate the model.
    int eval_iters; // How many random batches to evaluate over.
    bool final;     // Final training, using all data.
    float lr = 4e-5;  // Maximum learning rate for Adam optimizer.
    bool decay_lr = true;
    int warmup_iters = 150;
    int lr_decay_iters = 10000; // should be ~= total epochs
    float min_lr = 1e-7; // should be max_rate / 10
    bool train_ensemble = true; // train both hits and offset transformers
};

float get_lr(int ep, TrainConfig config) {
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


void getBatch(
    std::map<std::string, std::vector<torch::Tensor>>& data,
    int batch_size,
    int block_size,
    torch::Tensor& x,
    torch::Tensor& y
) {
    int num_samples = data["X"].size();
    if (num_samples == 0) {
        std::cerr << "Error: no data." << std::endl;
        return;
    }
    // std::cout << num_samples << " takes." << std::endl;
    torch::Tensor takes = torch::randint(0, num_samples, {batch_size});
    int take_ix = takes[0].item<int>();
    torch::Tensor start_ixs = torch::randint(0, 
        data["X"][take_ix].size(0) - block_size, {1});
    x = data["X"][take_ix].slice(
        0, start_ixs[0].item<int>(), start_ixs[0].item<int>() + block_size).
        unsqueeze(0);
    y = data["Y"][take_ix].slice(
        0, start_ixs[0].item<int>(), start_ixs[0].item<int>() + block_size).
        unsqueeze(0);

    for (int i = 1; i < batch_size; i++) {
        int take_ix = takes[i].item<int>();
        start_ixs = torch::cat(
            {start_ixs, torch::randint(0, data["X"][take_ix].size(0) - block_size, {1})});     
        x = torch::cat({x,
            data["X"][take_ix].slice(
                0, start_ixs[i].item<int>(), start_ixs[i].item<int>() + block_size).
                unsqueeze(0)
            });
        y = torch::cat({y,
            data["Y"][take_ix].slice(
                0, start_ixs[i].item<int>(), start_ixs[i
                ].item<int>() + block_size).
                unsqueeze(0)
            });
    }
    // std::cout << "take number: " << takes[0].item<int>() << std::endl;
    // std::cout << "start index: " << start_ixs[0].item<int>() << std::endl;
    if (x.size(2) >= 9) { // GMD data
        dataScaleDown(x);
        dataScaleDown(y);
    }
}

torch::Tensor computeLoss(torch::Tensor y_hat, torch::Tensor y) {
    // discard zero note offsets from y
    auto mask = (y.index({ Slice(), Slice(), Slice(0, 9) }) == 0.0);

    y_hat.narrow(2, 9, 9).index_put_(
        { mask.to(torch::kBool) }, 
        y_hat.narrow(2, 9, 9).index({ mask.to(torch::kBool) }) * 0.0);
    y_hat.narrow(2, 0, 9).index_put_(
        { mask.to(torch::kBool) }, 
        y_hat.narrow(2, 0, 9).index({ mask.to(torch::kBool) }) * 0.0);

    // predict tau_guitar to be the average non-zero offset
    torch::Tensor y_offsets = y.index({Slice(), Slice(), Slice(9, 18)});
    torch::Tensor non_zero_mask = (y_offsets != 0).to(torch::kFloat32);
    torch::Tensor non_zero_sum = (y_offsets * non_zero_mask).sum(2);
    torch::Tensor non_zero_count = non_zero_mask.sum(2);
    torch::Tensor mean_offsets = non_zero_sum / non_zero_count.clamp_min(1);

    y = y.slice(2, 0, y_hat.size(2) - 1);
    y = torch::cat({y, mean_offsets.unsqueeze(2) }, 2);

    return torch::mse_loss(y_hat, y); // vels, offsets, meanoffset-guitar
}

torch::Tensor hitsLoss(torch::Tensor y_hat, torch::Tensor y) {
    torch::Tensor y_hat_hits = y_hat.index({Slice(), Slice(), Slice(1,10)});
    torch::Tensor y_hits = threshToOnes(y).to(torch::kFloat);

    torch::Tensor y_hat_pos = y_hat.index({Slice(), Slice(), 0});
    torch::Tensor y_pos = torch::frac(y.index({Slice(), Slice(), INX_BAR_POS}));

    // std::cout << y_hits.sizes() << " " << y_hat.sizes() << std::endl;

    return torch::cross_entropy_loss(y_hat_hits, y_hits) + torch::mse_loss(y_hat_pos, y_pos);
}

float estimateLoss(TransformerModel model,
            TrainConfig config,
            std::map<std::string, std::vector<torch::Tensor>>& val_data,
            torch::Device device = torch::kCPU) {

    model->eval();
    float eval_loss = 0.0;
    torch::Tensor losses = torch::zeros({config.eval_iters});
    for (int i = 0; i < config.eval_iters; i++) {
        torch::Tensor x, y;
        getBatch(val_data, config.batch_size, config.block_size, x, y);
        torch::Tensor y_hat = model->forward(x);
        torch::Tensor loss = computeLoss(y_hat, y);
        losses[i] = loss.item<float>();
    }
    eval_loss = losses.mean().item<float>();
    model->train();
    return eval_loss;
}

float estimateLoss(HitsTransformer model,
            TrainConfig config,
            std::map<std::string, std::vector<torch::Tensor>>& val_data,
            torch::Device device = torch::kCPU) {

    model->eval();
    float eval_loss = 0.0;
    torch::Tensor losses = torch::zeros({config.eval_iters});
    for (int i = 0; i < config.eval_iters; i++) {
        torch::Tensor x, y;
        getBatch(val_data, config.batch_size, config.block_size, x, y);
        torch::Tensor y_hat = model->forward(x);
        torch::Tensor loss = hitsLoss(y_hat, y);
        losses[i] = loss.item<float>();
    }
    eval_loss = losses.mean().item<float>();
    model->train();
    return eval_loss;
}

void train(HitsTransformer hitsModel,
            TransformerModel model,
            TrainConfig config,
            std::map<std::string, std::vector<torch::Tensor>>& train_data,
            std::map<std::string, std::vector<torch::Tensor>>& val_data,
            std::string save_hits_model = "out/hitsModel.pt",
            std::string save_model = "out/model.pt",
            torch::Device device = torch::kCPU) 
{    
    bool trainHits = false;
    if (hitsModel) {
        trainHits = true;
        std::cout << "Training Ensemble..." << std::endl;
    }
    else {
        hitsModel = HitsTransformer(1,1,1,device); // dummy
        std::cout << "Training Main Transformer..." << std::endl;
    }

    torch::optim::Adam hitsOptimizer(hitsModel->parameters(), torch::optim::AdamOptions(config.lr));
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(config.lr));
    double min_loss = std::numeric_limits<double>::infinity();
    double min_hit_loss = std::numeric_limits<double>::infinity();

    hitsModel->train();
    model->train();

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        optimizer.zero_grad();
        hitsOptimizer.zero_grad();

        float lr = get_lr(epoch, config);
        for (auto param_group : hitsOptimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(lr);
        }
        for (auto param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions &>(param_group.options()).lr(lr);
        } // set lr: https://stackoverflow.com/questions/62415285/updating-learning-rate-with-libtorch-1-5-and-optimiser-options-in-c

        torch::Tensor x, y;
        getBatch(train_data, 
            config.batch_size, 
            config.block_size,
            x, y);

        torch::Tensor x_hits, loss;

        if (trainHits) {
            x_hits = hitsModel->forward(x);
            loss = hitsLoss(x_hits, y);
            loss.backward();
            nn::utils::clip_grad_norm_(hitsModel->parameters(), 0.5);
            hitsOptimizer.step();
            // update encoder input with predicted hits
            x.index_put_({Slice(), Slice(), Slice(0, 9)},
                x_hits.index({Slice(), Slice(), Slice(1, 10)})
                );
            x.detach_();
        }
                
        torch::Tensor out = model->forward(x);
        loss = computeLoss(out, y);
        loss.backward();
        nn::utils::clip_grad_norm_(model->parameters(), 0.5);
        optimizer.step();

        if (epoch % config.eval_interval == 0) {
            if (trainHits) {
                float eval_loss = estimateLoss(hitsModel, config, val_data, device);  
                if (eval_loss < min_hit_loss) {
                    min_hit_loss = eval_loss;
                    std::cout << "New min val hit_loss: " << min_hit_loss << std::endl;
                    // Save the model checkpoint.
                    torch::save(hitsModel, save_hits_model);
                }                
            }
            float eval_loss = estimateLoss(model, config, val_data, device);  
            if (eval_loss < min_loss) {
                min_loss = eval_loss;
                std::cout << "New min val loss: " << min_loss << std::endl;
                // Save the model checkpoint.
                torch::save(model, save_model);
            }
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " - train loss: " << loss.item<float>() << " | lr: " << lr << std::endl;
        }
    }
}

void finetune(TransformerModel model, 
                TrainConfig config,
                at::Tensor score,
                std::vector<std::array<double, INPUT_DIM>> play_notes,
                bool m_follow,
                torch::Device device = torch::kCPU)
{
    
}

} // end namespace backend