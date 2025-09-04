#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

__device__ void AddRoundKey(state_t* state, uint8_t round, const uint32_t* roundKey) {
	size_t offset = (size_t)round * 4;
	const uint4* roundKeyVec = reinterpret_cast<const uint4*>(roundKey + offset);

	// Reinterpret state as uint32_t for efficient word-wise operations
	uint4* stateVec = reinterpret_cast<uint4*>(*state);

	// Get lane ID within the warp
	unsigned int lane = threadIdx.x % 32;

	uint4 rk = { 0, 0, 0, 0 };
	if (lane == 0) {
		rk = *roundKeyVec;
	}

	// XOR 16 bytes in one instruction group
	stateVec[0].x ^= __shfl_sync(0xFFFFFFFF, rk.x, 0);
	stateVec[0].y ^= __shfl_sync(0xFFFFFFFF, rk.y, 0);
	stateVec[0].z ^= __shfl_sync(0xFFFFFFFF, rk.z, 0);
	stateVec[0].w ^= __shfl_sync(0xFFFFFFFF, rk.w, 0);
}