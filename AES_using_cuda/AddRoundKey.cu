#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

__device__ void AddRoundKey(state_t* state, uint8_t round, const uint8_t* roundKey) {
	int tid = threadIdx.x;
	int row = tid & 3;
	int col = tid >> 2;

	if (tid < 16) {
		uint32_t index = (round * Nb * 4) + (col * Nb) + row;
		if (index >= AES_EXPANDED_KEY_SIZE) {
			printf("Out of bounds access in AddRoundKey at index %u\n", index);
			return;
		}
		else {
			(*state)[row][col] ^= roundKey[index];
		}
	}
}