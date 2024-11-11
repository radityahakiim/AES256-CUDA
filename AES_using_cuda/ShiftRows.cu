#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

__global__ void ShiftRowsKernel(uint8_t* state) {
	int idx = threadIdx.x;

	if (idx == 1) {
		// Second row shifts one position to left
		uint8_t temp = state[1];
		state[1]     = state[5];
		state[5]     = state[9];
		state[9]     = state[13];
		state[13]    = temp;
	}
	else if (idx == 2) {
		// Third row shift two positions to the left
		uint8_t temp1 = state[2];
		uint8_t temp2 = state[6];
		state[2]      = state[10];
		state[6]      = state[14];
		state[10]     = temp1;
		state[14]     = temp2;
	}
	else if (idx == 3) {
		uint8_t temp = state[3];
		state[3]     = state[15];
		state[15]    = state[11];
		state[11]    = state[7];
		state[7]     = temp;
	}
}