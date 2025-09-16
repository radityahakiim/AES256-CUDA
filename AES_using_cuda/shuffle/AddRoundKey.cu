#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

__device__ __forceinline__ uint32_t warp_rk_lookup(int wordIdx, const uint32_t local_rk[2]) {
	int ownerLane = wordIdx >> 1;
	int offset    = wordIdx &  1;
	unsigned mask = __activemask();
	int v = static_cast<int>(local_rk[offset]);
	return static_cast<uint32_t>(__shfl_sync(mask, v, ownerLane));
}

__device__ void AddRoundKey(state_t* state, uint8_t round, const uint32_t local_rk[2]) {
	uint32_t* sWords = reinterpret_cast<uint32_t*>(*state); // Treat 16-byte AES state as 4 words
	int baseIdx = static_cast<int>(round) * 4;
	int lane = threadIdx.x & 3;
	uint32_t rkWord = warp_rk_lookup(baseIdx + lane, local_rk);
	sWords[lane] ^= rkWord;
}