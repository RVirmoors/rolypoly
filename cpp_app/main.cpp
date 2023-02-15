#include "torch/torch.h"
#include "torch/script.h"
#include <iostream>
#include <chrono>

using namespace std;

int main() {
    torch::jit::script::Module mod;
    std::string m_path = "c:/Users/rv/Documents/Max 8/Packages/rolypoly/help/roly.ts";

    try {
        mod = torch::jit::load(m_path);
        mod.eval();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
    }

    torch::Tensor output;

    for (int i = 0; i < 15; i++) { 
        torch::Tensor input = torch::rand({1, 5, 1});
        auto start = std::chrono::high_resolution_clock::now();
        output = mod.forward({input}).toTensor().detach();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        cout << output[0][0][0] << endl;
        cout << "duration: " << duration.count() / 1000. << " ms" << endl;
    }
}