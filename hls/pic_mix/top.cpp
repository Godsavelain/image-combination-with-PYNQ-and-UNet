#include <string.h>
#include "top.h"

void top(hls::stream<axis_t>& pic_input, hls::stream<axis_t>& mask_input,  hls::stream<axis_t>& back_input, hls::stream<axis_t>& output, int& sum ,int& num)
{
	#pragma HLS INTERFACE axis port=pic_input
	#pragma HLS INTERFACE axis port=back_input
	#pragma HLS INTERFACE axis port=mask_input
	#pragma HLS INTERFACE axis port=output
	#pragma HLS INTERFACE s_axilite port=sum
	#pragma HLS INTERFACE s_axilite port=num
	#pragma HLS INTERFACE s_axilite port=return

	loop_height: for (int i = 0; i < 600; ++i) {
		#pragma HLS LOOP_TRIPCOUNT max=600
		loop_width: for (int j = 0; j < 800; ++j) {
			#pragma HLS LOOP_TRIPCOUNT max=800
			#pragma HLS pipeline
			axis_t picb = pic_input.read();
			axis_t picg = pic_input.read();
			axis_t picr = pic_input.read();

			axis_t backb = back_input.read();
			axis_t backg = back_input.read();
			axis_t backr = back_input.read();

			axis_t maskb = mask_input.read();
			axis_t maskg = mask_input.read();
			axis_t maskr = mask_input.read();

			u1 skin_region;
			u8 r, g, b;
			u8 r_back, g_back, b_back;
			u8 r_mask, g_mask, b_mask;
		/***********stream pic_input *********/
			b = picb.data;
			g = picg.data;
			r = picr.data;

			b_back = backb.data;
			g_back = backg.data;
			r_back = backr.data;

			b_mask = maskb.data;
			g_mask = maskg.data;
			r_mask = maskr.data;

		/********get output*******/
			bool mask = ((b_mask==(u8)255) && (g_mask==(u8)255) && (r_mask==(u8)255));

			axis_t tmp1;
			axis_t tmp2;
			axis_t tmp3;
			tmp1.data = mask ? b : b_back;
			tmp1.last = 0;
			tmp2.data = mask ? g : g_back;
			tmp2.last = 0;
			tmp3.data = mask ? r : r_back;
			tmp3.last = ((i == 599) && (j == 799)) ? 1 : 0;
			output.write(tmp1);
			output.write(tmp2);
			output.write(tmp3);
		}
	}

}
