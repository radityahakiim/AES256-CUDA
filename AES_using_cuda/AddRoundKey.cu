#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

__device__ void AddRoundKey(state_t* state, uint8_t round, const uint32_t* roundKey) {
	size_t offset = (size_t)round * 4;
	const uint4* roundKeyVec = reinterpret_cast<const uint4*>(roundKey + offset);

	// Reinterpret state as uint32_t for efficient word-wise operations
	uint4* stateVec = reinterpret_cast<uint4*>(*state);

	// XOR 16 bytes in one instruction group
	stateVec[0].x ^= roundKeyVec[0].x;
	stateVec[0].y ^= roundKeyVec[0].y;
	stateVec[0].z ^= roundKeyVec[0].z;
	stateVec[0].w ^= roundKeyVec[0].w;
}