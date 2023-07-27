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

torch::Tensor threshToOnes(torch::Tensor src, float thresh = 0.0) {
    return torch::where(
                src.index({Slice(), Slice(), Slice(0, 9)}) > thresh,
                1.0,
                0.0
            ); // replace all hits with 1
}

struct HitsTransformerImpl : nn::Module {
// predicting upcoming hits
    HitsTransformerImpl(int d_model, int nhead, int enc_layers, torch::Device device) :
    device(device),
    d_model(d_model),
    pos_linLayer(nn::Linear(3, d_model)),
    hitsEmbedding(nn::Embedding(512, d_model)), // 2^9 possible hit combinations
    hitsTransformer(nn::TransformerEncoder(nn::TransformerEncoderOptions(nn::TransformerEncoderLayerOptions(d_model, nhead), enc_layers))),
    masker(nn::Transformer(nn::TransformerOptions())), // just to generate mask
    hitsFc(nn::Linear(d_model, ENCODER_DIM))    
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

    torch::Tensor oneHotToInt(torch::Tensor src) {
        // convert one hot to ints
        torch::Tensor mask = torch::pow(2, torch::arange(9, device = src.device()));
        return torch::sum(src * mask, 2).to(torch::kInt32);
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt /*not used*/) {
        torch::Tensor pos = src.index({Slice(), Slice(), Slice(INX_BPM, INX_TAU_G)});
        
        src = threshToOnes(src);
        src = oneHotToInt(src).to(device);// torch::cat({src, pos}, 2);
        
        torch::Tensor src_posenc = generatePE(pos);
        src = hitsEmbedding(src) * sqrt(d_model) + src_posenc;
        torch::Tensor src_mask = masker->generate_square_subsequent_mask(src.size(1)).to(device);
            
        src.transpose_(0, 1);    // (B, T, C) -> (T, B, C)
        torch::Tensor output = hitsTransformer(src, src_mask);
        output.transpose_(0, 1); // (T, B, C) -> (B, T, C)

        output = hitsFc(output);
        return output;
    }

    nn::Embedding hitsEmbedding;
    nn::TransformerEncoder hitsTransformer;
    nn::Transformer masker; // just to generate mask
    nn::Linear pos_linLayer, hitsFc;
    torch::Device device;
    double d_model;
};
TORCH_MODULE(HitsTransformer);

struct TransformerModelImpl : nn::Module {
    TransformerModelImpl(int input_dim, int output_dim, int d_model, int nhead, int enc_layers, int dec_layers, torch::Device device) :
    device(device),
    pos_linLayer(nn::Linear(3, d_model)),
    embedding(nn::Linear(input_dim, d_model)),
    embeddingEnc(nn::Linear(9, d_model)),
    transformer(nn::Transformer(nn::TransformerOptions(d_model, nhead, enc_layers, dec_layers))),
    fc(nn::Linear(d_model, output_dim))
    {   
        register_module("pos_linLayer", pos_linLayer);
        register_module("embedding", embedding);
        register_module("embeddingEnc", embeddingEnc);
        register_module("transformer", transformer);
        register_module("fc", fc);
        pos_linLayer->to(device);
        embedding->to(device);
        embeddingEnc->to(device);
        transformer->to(device);
        fc->to(device);
    }

    torch::Tensor generatePE(torch::Tensor x) {
        return pos_linLayer(x.index({Slice(), Slice(), Slice(INX_BPM, INX_TAU_G)}));
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt) {
        torch::Tensor src_posenc = generatePE(src);
        torch::Tensor tgt_posenc = generatePE(tgt);

        src = threshToOnes(src);
        std::cout << "SRC: " << src[0][0] << std::endl;
        std::cout << "TGT: " << tgt[0][0] << std::endl;
        std::cin.get();

        src = embeddingEnc(src) + src_posenc;
        tgt = embedding(tgt) + tgt_posenc;

        torch::Tensor src_mask = torch::zeros({src.size(1),src.size(1)}).to(device);
        torch::Tensor tgt_mask = transformer->generate_square_subsequent_mask(tgt.size(1)).to(device);
            
        // (B, T, C) -> (T, B, C)
        src.transpose_(0, 1);
        tgt.transpose_(0, 1);

        torch::Tensor output = transformer(src, tgt, src_mask, tgt_mask);
        
        output.transpose_(0, 1); // (T, B, C) -> (B, T, C)
        
        output = fc(output);
        return output;
    }

    nn::Linear pos_linLayer, embedding, embeddingEnc, fc;
    nn::Transformer transformer;
    torch::Device device;
};
TORCH_MODULE(TransformerModel);

} // end namespace backend