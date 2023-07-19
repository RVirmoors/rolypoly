#include <iostream>
#include <torch/torch.h>

namespace backend {

using namespace torch;

void train(TransformerModel model,
            torch::Tensor data,
            torch::Tensor input_seq,
            torch::Tensor output,
            std::string save_model = "model.pt",
            torch::Device device = torch::kCPU) 
{
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(4e-5));
    double min_loss = std::numeric_limits<double>::infinity();

    std::cout << "Training..." << std::endl;
    model->train();
    for (int epoch = 0; epoch < 500; epoch++) {
        double total_loss = 0.0;
        for (size_t i = 0; i < input_seq.size(0); i++) {
            optimizer.zero_grad();

            // Assuming src and tgt are torch::Tensor inputs
            torch::Tensor src = data[i];
            torch::Tensor tgt = input_seq[i];

            torch::Tensor out = model->forward(src, tgt);
            torch::Tensor loss = torch::mse_loss(out, output[i]);

            loss.backward();
            optimizer.step();

            total_loss += loss.item<double>();
        }

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

    std::cout << model->forward(data[0], input_seq[0]) << std::endl;
}

} // end namespace backend