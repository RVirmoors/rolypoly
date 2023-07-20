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
    bool final;     // Final training, using all data.
};

void dataScaleDown(torch::Tensor& data) {
    /*
    Scale the input data to range [0, 1].

    9 velocities from [0, 127]
    9 offsets from [-0.04, 0.04]
    bpm from [40, 240]
    bar_pos keep fractional part

    input: (batch, block_size, input_dim)
    output: (batch, block_size, input_dim)
    */
   data.index({Slice(), Slice(), Slice(0, 9)}).div_(127);
   data.index({Slice(), Slice(), Slice(9, 18)}).div_(0.08);
   data.index({Slice(), Slice(), Slice(INX_BPM, INX_BPM + 1)}).sub_(40).div_(200);
   data.index({Slice(), Slice(), Slice(INX_BAR_POS, INX_BAR_POS + 1)}).frac_();
}

void getBatch(
    std::map<std::string, std::vector<torch::Tensor>>& data,
    int batch_size,
    int block_size,
    torch::Tensor& x_enc,
    torch::Tensor& x_dec,
    torch::Tensor& y
) {
    int num_samples = data["X_enc"].size();
    if (num_samples == 0) {
        std::cerr << "Error: no data." << std::endl;
        return;
    }
    std::cout << num_samples << " takes." << std::endl;
    torch::Tensor takes = torch::randint(0, num_samples, {batch_size});
    int take_ix = takes[0].item<int>();
    torch::Tensor start_ixs = torch::randint(0, 
        data["X_enc"][take_ix].size(0) - block_size, {1});
    x_enc = data["X_enc"][take_ix].slice(
        0, start_ixs[0].item<int>(), start_ixs[0].item<int>() + block_size).
        unsqueeze(0);
    x_dec = data["X_dec"][take_ix].slice(
        0, start_ixs[0].item<int>(), start_ixs[0].item<int>() + block_size).
        unsqueeze(0);
    y = data["Y"][take_ix].slice(
        0, start_ixs[0].item<int>(), start_ixs[0].item<int>() + block_size).
        unsqueeze(0);

    for (int i = 1; i < batch_size; i++) {
        int take_ix = takes[i].item<int>();
        start_ixs = torch::cat(
            {start_ixs, torch::randint(0, data["X_enc"][take_ix].size(0) - block_size, {1})});     
        x_enc = torch::cat({x_enc,
            data["X_enc"][take_ix].slice(
                0, start_ixs[i].item<int>(), start_ixs[i].item<int>() + block_size).
                unsqueeze(0)
            });
        x_dec = torch::cat({x_dec,
            data["X_dec"][take_ix].slice(
                0, start_ixs[i].item<int>(), start_ixs[i].item<int>() + block_size).
                unsqueeze(0)
            });
        y     = torch::cat({y,
            data["Y"][take_ix].slice(
                0, start_ixs[i].item<int>(), start_ixs[i
                ].item<int>() + block_size).
                unsqueeze(0)
            });
    }
    std::cout << x_enc.sizes();
    dataScaleDown(x_enc);
    dataScaleDown(x_dec);
    dataScaleDown(y);
}


void train(TransformerModel model,
            TrainConfig config,
            std::map<std::string, std::vector<torch::Tensor>>& train_data,
            std::map<std::string, std::vector<torch::Tensor>>& val_data,
            std::string save_model = "model.pt",
            torch::Device device = torch::kCPU) 
{
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(4e-5));
    double min_loss = std::numeric_limits<double>::infinity();

    std::cout << "Training..." << std::endl;
    model->train();

    for (int epoch = 0; epoch < 500; epoch++) {
        double total_loss = 0.0;
        optimizer.zero_grad();

        // Assuming src and tgt are torch::Tensor inputs
        torch::Tensor x_enc, x_dec, y;
        getBatch(train_data, 
            config.batch_size, 
            config.block_size,
            x_enc, x_dec, y);
        
        return;
        torch::Tensor y_hat = model->forward(x_enc, x_dec);
        torch::Tensor loss = torch::mse_loss(y_hat, y);

        loss.backward();
        optimizer.step();

        total_loss = loss.item<double>();
    

        if (total_loss < min_loss) {
            min_loss = total_loss;
            std::cout << "New min loss: " << min_loss << std::endl;
            // Save the model checkpoint.
            torch::save(model, save_model);
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " - " << total_loss << std::endl;
        }

    }

}

} // end namespace backend