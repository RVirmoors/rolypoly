/**
@file
rolypoly~.h: header
created by: grigore burloiu, gburloiu@gmail.com
*/

#pragma once

#include "torch\torch.h"
#include "ext.h"			// standard Max include, always required (except in Jitter)
#include "ext_obex.h"		// required for "new" style objects
#include "z_dsp.h"			// required for MSP objects


class Rolypoly;

extern "C" {
	// struct to represent the object's state
	typedef struct _rolypoly {
		t_pxobject		ob;			// the object itself (t_pxobject in MSP instead of t_object)
		Rolypoly*	roly;

		void* out_sig;	// outlets must be defined
	} t_rolypoly;

	// method prototypes
	void *rolypoly_new(t_symbol *s, long argc, t_atom *argv);
	void rolypoly_free(t_rolypoly *x);
	void rolypoly_assist(t_rolypoly *x, void *b, long m, long a, char *s);
	void rolypoly_float(t_rolypoly *x, double f);
	void rolypoly_dsp64(t_rolypoly *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
	void rolypoly_perform64(t_rolypoly *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);

	// global class pointer variable
	static t_class *rolypoly_class = NULL;
} // end extern "C"

class Rolypoly {
public:
	t_rolypoly *max;

	// internal vars:
	float offset;
	torch::Tensor tensor;

	// constructor & destructor:
	Rolypoly();
	~Rolypoly();

	// DSP:
	void perform(double *in, double *out, long sampleframes);

	// inlet methods:
	void in_float(double f);
};
