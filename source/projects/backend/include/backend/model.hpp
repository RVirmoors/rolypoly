#include <iostream>
#include <torch/torch.h>

namespace backend {

using namespace torch;

// TODO normalize all features

struct TransformerModelImpl : nn::Module {
    TransformerModelImpl(int input_dim, int output_dim, int d_model, int nhead, torch::Device device) :
    device(device),
    pos_linLayer(nn::Linear(input_dim, d_model)),
    embedding(nn::Linear(input_dim, d_model)),
    transformer(nn::Transformer(nn::TransformerOptions(d_model, nhead))),
    fc(nn::Linear(d_model, output_dim))
    {   
        register_module("pos_linLayer", pos_linLayer);
        register_module("embedding", embedding);
        register_module("transformer", transformer);
        register_module("fc", fc);
        pos_linLayer->to(device);
        embedding->to(device);
        transformer->to(device);
        fc->to(device);
    }

    torch::Tensor generatePE(torch::Tensor x) {
        return pos_linLayer(x);
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt) {
        torch::Tensor src_posenc = generatePE(src);
        torch::Tensor tgt_posenc = generatePE(tgt);
        src = embedding(src) + src_posenc;
        tgt = embedding(tgt) + tgt_posenc;
        torch::Tensor src_mask = transformer->generate_square_subsequent_mask(src.size(0)).to(device);
        src = src.unsqueeze(1); // add batch dimension, needed for transformer
        tgt = tgt.unsqueeze(1); // expects (T, B, C)
        torch::Tensor output = transformer(src, tgt, src_mask);
        output = output.squeeze(1); // remove batch dimension
        output = fc(output);
        return output;
    }

    nn::Linear pos_linLayer, embedding, fc;
    nn::Transformer transformer;
    torch::Device device;
};
TORCH_MODULE(TransformerModel);

} // end namespace backend