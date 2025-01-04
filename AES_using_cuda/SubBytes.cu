#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

// SubBytes (for encryption)
__device__ void SubBytes(state_t* state) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			(*state)[j][i] = getSBoxValueDevice((*state)[j][i]);
		}
	}
}

// Inverse SubBytes (for decryption)
__device__ void InvSubBytes(state_t* state) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			(*state)[j][i] = getSBoxInvertDevice((*state)[j][i]);
		}
	}
}