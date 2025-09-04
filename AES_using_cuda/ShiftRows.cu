#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

__device__ inline uint32_t rotl32(uint32_t v, int shift_bytes) {
	// rotate left by (shift_bytes * 8) bits
	int s = (shift_bytes & 3) * 8;
	return (v << s) | (v >> (32 - s));
}

__device__ void ShiftRows(state_t* state) {
	// Pack each row (4 bytes across columns 0..3) into a 32-bit word.
	// Layout: byte0 = column 0, byte1 = column1, byte2 = column2, byte3 = column3
	uint32_t r0 = (uint32_t)(*state)[0][0] | ((uint32_t)(*state)[0][1] << 8) |
		((uint32_t)(*state)[0][2] << 16) | ((uint32_t)(*state)[0][3] << 24);

	uint32_t r1 = (uint32_t)(*state)[1][0] | ((uint32_t)(*state)[1][1] << 8) |
		((uint32_t)(*state)[1][2] << 16) | ((uint32_t)(*state)[1][3] << 24);

	uint32_t r2 = (uint32_t)(*state)[2][0] | ((uint32_t)(*state)[2][1] << 8) |
		((uint32_t)(*state)[2][2] << 16) | ((uint32_t)(*state)[2][3] << 24);

	uint32_t r3 = (uint32_t)(*state)[3][0] | ((uint32_t)(*state)[3][1] << 8) |
		((uint32_t)(*state)[3][2] << 16) | ((uint32_t)(*state)[3][3] << 24);

	const unsigned warp_mask = 0xFFFFFFFFu;
	int lane_id = threadIdx.x & 31; // lane within warp
	// Example: rotate each rowword among lanes by (shift_bytes) lanes (NOT generally useful unless layout matches).
	// We still compute the per-row byte-rotations locally to guarantee correctness if the shuffle is not applicable.
	uint32_t r1_rot_local = rotl32(r1, 1); // shift 1 byte left
	uint32_t r2_rot_local = rotl32(r2, 2); // shift 2 bytes left
	uint32_t r3_rot_local = rotl32(r3, 3); // shift 3 bytes left

	// The following is a no-op if lane_id used as dest == src; it's here to show warp primitive usage.
	// If you have a meaningful mapping you can change the srcLane indices appropriately.
	uint32_t r1_rot = __shfl_sync(warp_mask, r1_rot_local, lane_id);
	uint32_t r2_rot = __shfl_sync(warp_mask, r2_rot_local, lane_id);
	uint32_t r3_rot = __shfl_sync(warp_mask, r3_rot_local, lane_id);
	uint32_t r0_rot = r0; // row 0 is unchanged

	// Unpack rotated bytes back into state[row][col]
	(*state)[0][0] = (uint8_t)(r0_rot & 0xFF);
	(*state)[0][1] = (uint8_t)((r0_rot >> 8) & 0xFF);
	(*state)[0][2] = (uint8_t)((r0_rot >> 16) & 0xFF);
	(*state)[0][3] = (uint8_t)((r0_rot >> 24) & 0xFF);

	(*state)[1][0] = (uint8_t)(r1_rot & 0xFF);
	(*state)[1][1] = (uint8_t)((r1_rot >> 8) & 0xFF);
	(*state)[1][2] = (uint8_t)((r1_rot >> 16) & 0xFF);
	(*state)[1][3] = (uint8_t)((r1_rot >> 24) & 0xFF);

	(*state)[2][0] = (uint8_t)(r2_rot & 0xFF);
	(*state)[2][1] = (uint8_t)((r2_rot >> 8) & 0xFF);
	(*state)[2][2] = (uint8_t)((r2_rot >> 16) & 0xFF);
	(*state)[2][3] = (uint8_t)((r2_rot >> 24) & 0xFF);

	(*state)[3][0] = (uint8_t)(r3_rot & 0xFF);
	(*state)[3][1] = (uint8_t)((r3_rot >> 8) & 0xFF);
	(*state)[3][2] = (uint8_t)((r3_rot >> 16) & 0xFF);
	(*state)[3][3] = (uint8_t)((r3_rot >> 24) & 0xFF);
}

__device__ void InvShiftRows(state_t* state) {
	// Second row shifts right by 1 position
	uint8_t temp = (*state)[3][1];
	(*state)[3][1] = (*state)[2][1];
	(*state)[2][1] = (*state)[1][1];
	(*state)[1][1] = (*state)[0][1];
	(*state)[0][1] = temp;
	// Third row shifts right by 2 position
	temp = (*state)[0][2];
	(*state)[0][2] = (*state)[2][2];
	(*state)[2][2] = temp;
	temp = (*state)[1][2];
	(*state)[1][2] = (*state)[3][2];
	(*state)[3][2] = temp;
	// Fourth row shifts right by 3 position
	temp = (*state)[0][3];
	(*state)[0][3] = (*state)[1][3];
	(*state)[1][3] = (*state)[2][3];
	(*state)[2][3] = (*state)[3][3];
	(*state)[3][3] = temp;
}