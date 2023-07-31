// Rolypoly C++ implementation
// 2023 rvirmoors
//
// Backend: model definition and forward pass

#pragma once

#include <iostream>
#include <cmath>
#include <torch/torch.h>

namespace backend {

using namespace torch;
using namespace at::indexing;

torch::Tensor threshToOnes(torch::Tensor src, float startIndex = 9, float thresh = 0.0) {
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

        // std::cout << "SRC: " << src[0][0] << std::endl;
        // std::cout << "TGT: " << tgt[0][0] << std::endl;
        // std::cin.get();

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


struct HitsTransformerImpl : nn::Module {
// predicting upcoming hits
    HitsTransformerImpl(int d_model, int nhead, int enc_layers, torch::Device device) :
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
        src = threshToOnes(src).to(device);

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
TORCH_MODULE(HitsTransformer);

} // end namespace backend