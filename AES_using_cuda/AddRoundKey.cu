#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

__device__ void AddRoundKey(uint8_t* state, const uint8_t* roundKey, int blockSize) {
	for (int i = 0; i < blockSize; i++) {
		state[i] ^= roundKey[i];
	}
}