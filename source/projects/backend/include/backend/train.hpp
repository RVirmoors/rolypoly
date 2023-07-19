#include <iostream>
#include <torch/torch.h>

namespace backend {

using namespace torch;

void train() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA." << std::endl;
        device = torch::kCUDA;
    }

    TransformerModel model(5, 5, 64, 8, device);

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

    // // Print the tensors for verification (optional).
    // std::cout << "input_seq:" << std::endl << input_seq << std::endl;
    // std::cout << "output:" << std::endl << output << std::endl;
    // std::cout << "data:" << std::endl << data << std::endl;

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
            //torch::save({{"model", model}, {"optimizer", optimizer}}, load_model);
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " - " << total_loss << std::endl;
        }
    }

    std::cout << model->forward(data[0], input_seq[0]) << std::endl;
}

} // end namespace backend