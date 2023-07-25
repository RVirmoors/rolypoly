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

struct HitsTransformerImpl : nn::Module {
// predicting upcoming hits
    HitsTransformerImpl(int d_model, int nhead, int enc_layers, torch::Device device) :
    device(device),
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
        torch::Tensor output = torch::zeros({src.size(0), src.size(1), 1},
            torch::dtype(torch::kInt32));
        for (int i = 0; i < src.size(0); i++) {
            for (int j = 0; j < src.size(1); j++) {
                for (int k = 0; k < src.size(2); k++) {
                    if (src[i][j][k].item<float>() == 1.0) {
                        output[i][j][0] += (int)std::pow(2, k);
                    }
                }
            }
        }
        return output;
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt) {
        torch::Tensor pos = src.index({Slice(), Slice(), Slice(INX_BPM, INX_TAU_G)});

        src = torch::where(
            src.index({Slice(), Slice(), Slice(0, 9)}) > 0,
            1.0,
            0.0
        ); // replace all hits with 1

        src = oneHotToInt(src).squeeze(2).to(device);// torch::cat({src, pos}, 2);
        
        torch::Tensor src_posenc = generatePE(pos);
        src = hitsEmbedding(src) + src_posenc;

        torch::Tensor src_mask = masker->generate_square_subsequent_mask(src.size(1)).to(device);
            
        src.transpose_(0, 1);    // (B, T, C) -> (T, B, C)
        torch::Tensor output = hitsTransformer(src, src_mask);
        output.transpose_(0, 1); // (T, B, C) -> (B, T, C)

        output = hitsFc(output);

        std::cout << output << output.sizes() << std::endl;
        std::cin.get();

        return output;
    }

    nn::Embedding hitsEmbedding;
    nn::TransformerEncoder hitsTransformer;
    nn::Transformer masker; // just to generate mask
    nn::Linear pos_linLayer, hitsFc;
    torch::Device device;
};
TORCH_MODULE(HitsTransformer);

struct TransformerModelImpl : nn::Module {
    TransformerModelImpl(int input_dim, int output_dim, int d_model, int nhead, int enc_layers, int dec_layers, torch::Device device) :
    device(device),
    pos_linLayer(nn::Linear(3, d_model)),
    embedding(nn::Linear(input_dim, d_model)),
    transformer(nn::Transformer(nn::TransformerOptions(d_model, nhead, enc_layers, dec_layers))),
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
        return pos_linLayer(x.index({Slice(), Slice(), Slice(INX_BPM, INX_TAU_G)}));
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

        torch::Tensor src_mask = torch::zeros({src.size(1),src.size(1)}).to(device);
        torch::Tensor tgt_mask = transformer->generate_square_subsequent_mask(tgt.size(1)).to(device);
            
        // (B, T, C) -> (T, B, C)
        src.transpose_(0, 1);
        tgt.transpose_(0, 1);

        torch::Tensor output = transformer(src, tgt, src_mask, tgt_mask);

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