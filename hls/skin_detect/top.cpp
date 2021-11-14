#include "imgprocess.h"
#include <string.h>

void ImgProcess_Top(hls::stream<axis_t>& input, hls::stream<axis_t>& output, int& sum ,int& num)
{
//	#pragma HLS RESOURCE variable=input  core=AXIS metadata="-bus_bundle INPUT_STREAM"
//	#pragma HLS RESOURCE variable=output core=AXIS metadata="-bus_bundle OUTPUT_STREAM"
//	#pragma HLS RESOURCE variable=sum    core=AXIS metadata="-bus_bundle OUTPUT_STREAM"
//	#pragma HLS RESOURCE variable=num    core=AXIS metadata="-bus_bundle OUTPUT_STREAM"
//	#pragma HLS RESOURCE core=AXI_SLAVE variable=return metadata="-bus_bundle CONTROL_BUS"

	#pragma HLS INTERFACE axis port=input
	#pragma HLS INTERFACE axis port=output
	#pragma HLS INTERFACE s_axilite port=sum
	#pragma HLS INTERFACE s_axilite port=num
	#pragma HLS INTERFACE s_axilite port=return

//#pragma HLS INTERFACE s_axilite port=input
//#pragma HLS INTERFACE s_axilite port=output
//#pragma HLS INTERFACE s_axilite port=sum
//#pragma HLS INTERFACE s_axilite port=num
//#pragma HLS INTERFACE ap_ctrl_none port = return

	int y_lower = 0;
	int y_upper = 255;
	int cb_lower = 80;
	int cb_upper = 135;
	int cr_lower = 131;
	int cr_upper = 185;
	int sum1 = 0;
	int num1 = 0;

	loop_height: for (int i = 0; i < 600; ++i) {
		#pragma HLS LOOP_TRIPCOUNT max=600
		loop_width: for (int j = 0; j < 800; ++j) {
			#pragma HLS LOOP_TRIPCOUNT max=800
			#pragma HLS pipeline
			axis_t tmpb = input.read();
			axis_t tmpg = input.read();
			axis_t tmpr = input.read();

			u1 skin_region;
			u8 r, g, b;
		/***********stream input *********/
			b = tmpb.data;
			g = tmpg.data;
			r = tmpr.data;

		/********detect skin region*******/
			skin_region = rgb2ycbcr(b, g, r,y_lower,y_upper,cb_lower,cb_upper,cr_lower,cr_upper);

			sum1 = sum1 + j*(skin_region);
			num1 = num1 + skin_region;

			axis_t tmp1;
			axis_t tmp2;
			axis_t tmp3;
			tmp1.data = skin_region ? b : (u8)0;
			tmp1.last = 0;
			tmp2.data = skin_region ? g : (u8)0;
			tmp2.last = 0;
			tmp3.data = skin_region ? r : (u8)0;
			tmp3.last = ((i == 599) && (j == 799)) ? 1 : 0;
			output.write(tmp1);
			output.write(tmp2);
			output.write(tmp3);
		}
	}

	sum = sum1;
	num = num1;

	printf("和：%d\n",sum);
	printf("个数：%d\n",num);
	printf("位置：%d\n",(int)(sum/num));
}
