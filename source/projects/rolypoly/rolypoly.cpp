/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2018 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "../shared/signal_routing_objects.h"

class rolypoly : public signal_routing_base<rolypoly>, public sample_operator<2, 2> {
public:
	MIN_DESCRIPTION {"Expressive Drum Machine: read MIDI file and output drums"};
	MIN_TAGS {"MIDI, drums, routing"};
	MIN_AUTHOR {"Grigore Burloiu // rvirmoors"};
	MIN_RELATED {"nn~, antescofo~"};

	inlet<>  in1 {this, "(signal) In 1"};
	outlet<> out1 {this, "(signal) Left Output", "signal"};

	// constructor
	rolypoly() {

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
