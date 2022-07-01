/**
	@file
	rolypoly~:
	based on simplemsp by: jeremy bernstein, jeremy@bootsquad.com
	created by: grigore burloiu, gburloiu@gmail.com
*/

#include "rolypoly~.h"

extern "C" { // ugly C stuff, bridges SDK to C++ class(es) below.
	int C74_EXPORT main(void) {
		// object initialization, note the use of dsp_free for the freemethod, which is required
		// unless you need to free allocated memory, in which case you should call dsp_free from
		// your custom free function.
		t_class *c = class_new("rolypoly~", (method)rolypoly_new, (method)dsp_free, (long)sizeof(t_rolypoly), 0L, A_GIMME, 0);

		class_addmethod(c, (method)rolypoly_float, "float", A_FLOAT, 0);
		class_addmethod(c, (method)rolypoly_dsp64, "dsp64", A_CANT, 0);
		class_addmethod(c, (method)rolypoly_assist, "assist", A_CANT, 0);

		class_dspinit(c);
		class_register(CLASS_BOX, c);
		rolypoly_class = c;
	}


	void *rolypoly_new(t_symbol *s, long argc, t_atom *argv)
	{
		t_rolypoly *x = (t_rolypoly *)object_alloc(rolypoly_class);
		x->roly = new Rolypoly();

		if (x) {
			x->roly->max = x; // get the C++ object from the classic SDK's C object
			dsp_setup((t_pxobject *)x, 1);	// MSP inlets: arg is # of inlets and is REQUIRED!
			// use 0 if you don't need inlets
			// outlets, right to left:
			x->out_sig = outlet_new((t_object *) x, "signal"); 		// signal outlet (note "signal" rather than NULL)
		}
		return (x);
	}

	// NOT CALLED!, we use dsp_free for a generic free function
	void rolypoly_free(t_rolypoly *x)
	{
		;
	}

	void rolypoly_assist(t_rolypoly *x, void *b, long m, long a, char *s)
	{
		if (m == ASSIST_INLET) { //inlet
			sprintf(s, "I am inlet %ld", a);
		}
		else {	// outlet
			sprintf(s, "I am outlet %ld", a);
		}
	}


	void rolypoly_float(t_rolypoly *x, double f)
	{
		x->roly->in_float(f);			// just call the C++ method
	}

	// registers a function for the signal chain in Max
	void rolypoly_dsp64(t_rolypoly *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
	{
		post("my sample rate is: %f", samplerate);

		// instead of calling dsp_add(), we send the "dsp_add64" message to the object representing the dsp chain
		// the arguments passed are:
		// 1: the dsp64 object passed-in by the calling function
		// 2: the symbol of the "dsp_add64" message we are sending
		// 3: a pointer to your object
		// 4: a pointer to your 64-bit perform method
		// 5: flags to alter how the signal chain handles your object -- just pass 0
		// 6: a generic pointer that you can use to pass any additional data to your perform method

		object_method(dsp64, gensym("dsp_add64"), x, rolypoly_perform64, 0, NULL);
	}

	// this is the 64-bit perform method audio vectors
	void rolypoly_perform64(t_rolypoly *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
	{
		t_double *inL = ins[0];		// we get audio for each inlet of the object from the **ins argument
		t_double *outL = outs[0];	// we get audio for each outlet of the object from the **outs argument
		x->roly->perform(inL, outL, sampleframes);	// just call the C++ method
	}
} // end extern "C"

//*********************
// C++ code starts here
//*********************

Rolypoly::Rolypoly() {
	// init vars. These are declared in the .h file.
	offset = 0.0;
	tensor = torch::rand({ 2, 3 });
	// Max uses printf to post, we can't use iostream...
	post("random tensor: %.2f %.2f %.2f | %.2f %.2f %.2f ",
		tensor[0][0].item<float>(), tensor[0][1].item<float>(), tensor[0][2].item<float>(),
		tensor[1][0].item<float>(), tensor[1][1].item<float>(), tensor[1][2].item<float>());
}

Rolypoly::~Rolypoly() {
	// destructor (optional)
}

void Rolypoly::in_float(double f) {
	offset = (float)f;
}

void Rolypoly::perform(double *in, double *out, long sampleframes) {
	int n = sampleframes;
	// this perform method simply copies the input to the output, offsetting the value
	while (n--)
		*out++ = *in++ + offset;
}
