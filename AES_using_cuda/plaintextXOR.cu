#include <iostream>
#include <vector>
#include "aes_header.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 256

__global__ void XORWithKeyKernel(uint8_t* data, const uint8_t* key, int dataSize, int keySize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < dataSize) {
		data[idx] ^= key[idx % keySize];
	}
}