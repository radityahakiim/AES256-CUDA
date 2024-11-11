#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

__global__ void AddRoundKey(uint8_t* state, const uint8_t* roundKey) {
	int idx = threadIdx.x;
	if (idx < Nb * 4) {
		state[idx] ^= roundKey[idx];
	}
}