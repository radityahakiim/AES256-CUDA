#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

__device__ void AddRoundKey(state_t* state, uint8_t round, const uint32_t* roundKey) {
	size_t offset = (size_t)round * 4;
	const uint32_t* roundKeyStart = roundKey + offset;

	// Reinterpret state as uint32_t for efficient word-wise operations
	uint32_t* state_as_words = reinterpret_cast<uint32_t*>(*state);

#pragma unroll
	for (uint8_t i = 0; i < 4; i++) {
		state_as_words[i] ^= roundKeyStart[i];
	}
}