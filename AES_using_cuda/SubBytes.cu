#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

__global__ void SubBytesKernel(uint8_t* state) {
	int idx = threadIdx.x;

	if (idx < AES_BLOCK_SIZE) {
		state[idx] = sbox[state[idx]];
	}
}