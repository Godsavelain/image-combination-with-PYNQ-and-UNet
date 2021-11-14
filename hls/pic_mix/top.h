#include "hls_video.h"
#include <ap_fixed.h>

#define MAX_WIDTH  800
#define MAX_HEIGHT 600
typedef ap_uint<8> u8;
typedef ap_uint<1> u1;
typedef ap_uint<32> u32;

struct axis_t{
	u8 data;
    ap_int<1> last;
};

//template<class T>
//inline void axis_read(hls::stream<axis_t>& in_arr, T* out_arr, std::size_t size) {
//    for (int i = 0; i < size; ++i) {
//    #pragma HLS loop_tripcount max=307200
//        axis_t tmp = in_arr.read();
//        out_arr[i] = tmp.data;
//    }
//}
//
//template<class T>
//inline void axis_write(T* in_arr, hls::stream<axis_t>& out_arr, std::size_t size) {
//    for (int i = 0; i < size; ++i) {
//    #pragma HLS loop_tripcount max=307200
//        axis_t tmp;
//        tmp.data = in_arr[i];
//        if (i == size - 1)
//            tmp.last = 1; // be careful
//        else
//            tmp.last = 0;
//        out_arr.write(tmp);
//    }
//}

