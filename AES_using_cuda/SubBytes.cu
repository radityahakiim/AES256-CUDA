#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

// SubBytes (for encryption)
__device__ void SubBytes(state_t* state) {
	int idx = threadIdx.x;
	if (idx < 16) {
		int row = idx & 0x3;
		int col = idx >> 2;
		(*state)[row][col] = getSBoxValueDevice((*state)[row][col]);
	}
}

// Inverse SubBytes (for decryption)
__device__ void InvSubBytes(state_t* state) {
	int idx = threadIdx.x;
	if (idx < 16) {
		int row = idx & 0x3;
		int col = idx >> 2;
		(*state)[row][col] = getSBoxInvertDevice((*state)[row][col]);
	}
}