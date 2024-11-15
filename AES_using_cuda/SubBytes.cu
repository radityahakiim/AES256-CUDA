#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

__constant__ uint8_t d_sb[256];

__device__ void SubBytes(uint8_t* state) {
	for (int i = 0; i < AES_BLOCK_SIZE; i++) {
		state[i] = d_sb[state[i]];
	}
}