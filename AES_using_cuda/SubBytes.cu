#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

// SubBytes (for encryption)
__device__ void SubBytes(state_t* state) {
	int col = threadIdx.x % 4;
	int row = threadIdx.x / 4;

	if (threadIdx.x < 16) {
		(*state)[col][row] = getSBoxValueDevice((*state)[col][row]);
	}
}

// Inverse SubBytes (for decryption)
__device__ void InvSubBytes(state_t* state) {
	int col = threadIdx.x % 4;
	int row = threadIdx.x / 4;

	if (threadIdx.x < 16) {
		(*state)[col][row] = getSBoxInvertDevice((*state)[col][row]);
	}
}