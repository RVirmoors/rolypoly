// Rolypoly C++ implementation
// 2023 rvirmoors
//
// Backend: model definition and forward pass

#pragma once

#include <iostream>
#include <torch/torch.h>

namespace backend {

using namespace torch;
using namespace at::indexing;

struct TransformerModelImpl : nn::Module {
    TransformerModelImpl(int input_dim, int output_dim, int d_model, int nhead, torch::Device device) :
    device(device),
    pos_linLayer(nn::Linear(1, d_model)),
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
        return pos_linLayer(x.index({Slice(), Slice(), INX_BAR_POS}).unsqueeze(2));
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt) {
        bool noBatch = false;        
        if (src.sizes().size() == 2) {
            src.unsqueeze_(0); // add batch dimension, needed for transformer
            tgt.unsqueeze_(0); // (B, T, C)
            noBatch = true;
        }
        torch::Tensor src_posenc = generatePE(src);
        torch::Tensor tgt_posenc = generatePE(tgt);
        src = embedding(src) + src_posenc;
        tgt = embedding(tgt) + tgt_posenc;
        torch::Tensor src_mask = transformer->generate_square_subsequent_mask(src.size(1)).to(device);
        // (B, T, C) -> (T, B, C)
        src.transpose_(0, 1);
        tgt.transpose_(0, 1);

        torch::Tensor output = transformer(src, tgt, src_mask);
        if (noBatch)
            output.squeeze_(1); // remove batch dimension
        else 
            output.transpose_(0, 1); // (T, B, C) -> (B, T, C)
        
        output = fc(output);
        return output;
    }

    nn::Linear pos_linLayer, embedding, fc;
    nn::Transformer transformer;
    torch::Device device;
};
TORCH_MODULE(TransformerModel);

} // end namespace backend