#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

__device__ void AddRoundKey(state_t* state, uint8_t round, const uint8_t* roundKey) {
#pragma unroll
	for (uint8_t i = 0; i < 4; i++) {
#pragma unroll
		for (uint8_t j = 0; j < 4; j++) {
			uint32_t index = (round * Nb * 4) + (i * Nb) + j;
			if (index >= AES_EXPANDED_KEY_SIZE) {
				printf("Out of bounds access in AddRoundKey at index %u\n", index);
				return;
			}
			(*state)[i][j] ^= roundKey[index];
		}
	}
}