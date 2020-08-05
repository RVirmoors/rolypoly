inlets = 4

var d_g, g_d, dur;
d_g = g_d = 9999.;

function msg_float(v) {
	if(inlet == 0 && v < Math.abs(g_d)) {
		g_d = -v;
		}
	else if (inlet == 1 && v < d_g) {
		d_g = v;
		}
	else if (inlet == 2) {
		dur = v;
		}
	//post("\ng, d, dur:" + g_d + " " + d_g + " " + dur);
}

function bang() {
	if (inlet == 3) {
			var min;
			if (Math.abs(g_d) < d_g) {
				min = g_d;
				}
			else {
				min = d_g;	
				}
			if (Math.abs(min) < dur) {				
				outlet(0, min);
				}
			d_g = g_d = 9999.; // reset & wait for new diffs
		}
	}