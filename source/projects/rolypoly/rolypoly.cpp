/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "../shared/signal_routing_objects.h"
#include "torch\torch.h"

class rolypoly : public signal_routing_base<rolypoly>, public sample_operator<2, 2> {
public:
	MIN_DESCRIPTION {"Expressive Drum Machine"};
	MIN_TAGS {"audio, routing"};
	MIN_AUTHOR {"Grigore Burloiu // rvirmoors"};
	MIN_RELATED {"antescofo~"};

	inlet<>  in1 {this, "(signal) Input 1"};
	inlet<>  in_pos {this, "(signal) Position between them (0..1)"};
	outlet<> out1 {this, "(signal) Left Output", "signal"};
	outlet<> out2 {this, "(signal) Right Output", "signal"};

	// constructor
	rolypoly() {
		torch::Tensor tensor = torch::rand({ 2, 3 });
		cout << "random tensor: " 
			<< tensor[0][0].item<float>() << " " << tensor[0][1].item<float>() << " " << tensor[0][2].item<float>() 
			<< " | " 
			<< tensor[1][0].item<float>() << " " << tensor[1][1].item<float>() << " " << tensor[1][2].item<float>()
			<< endl;
	}


	/// Process one sample

	samples<2> operator()(sample input, sample position = 0.5) {
		auto weight1 = this->weight1;
		auto weight2 = this->weight2;

		if (in_pos.has_signal_connection())
			std::tie(weight1, weight2) = calculate_weights(mode, position);

		return { {input * weight1, input * weight2}};
	}
};

MIN_EXTERNAL(rolypoly);
