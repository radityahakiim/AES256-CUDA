#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

static __device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return ((x << 1) ^ ((x >> 7) & 0x1b));
}

__device__ __forceinline__ uint8_t mul09(uint8_t x) {
    return xtime(xtime(xtime(x))) ^ x;
}
__device__ __forceinline__ uint8_t mul0b(uint8_t x) {
    return xtime(xtime(xtime(x))) ^ xtime(x) ^ x;
}
__device__ __forceinline__ uint8_t mul0d(uint8_t x) {
    return xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ x;
}
__device__ __forceinline__ uint8_t mul0e(uint8_t x) {
    return xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ xtime(x);
}

__device__ void MixColumns(state_t* state) {
    int lane = threadIdx.x & 3;
    uint32_t* state32 = reinterpret_cast<uint32_t*>(*state);
    uint32_t col = state32[lane];

    // Extract bytes using a single mask and shifts
    uint8_t a0 = col & 0xFF;
    uint8_t a1 = (col >> 8) & 0xFF;
    uint8_t a2 = (col >> 16) & 0xFF;
    uint8_t a3 = (col >> 24) & 0xFF;
    
    // Compute XOR
    uint8_t t = a0 ^ a1 ^ a2 ^ a3;

    uint8_t x0 = xtime(a0 ^ a1);
    uint8_t x1 = xtime(a1 ^ a2);
    uint8_t x2 = xtime(a2 ^ a3);
    uint8_t x3 = xtime(a3 ^ a0);

    state32[lane] =
        ((uint32_t)(x3 ^ t ^ a3) << 24) |
        ((uint32_t)(x2 ^ t ^ a2) << 16) |
        ((uint32_t)(x1 ^ t ^ a1) << 8) |
        ((uint32_t)(x0 ^ t ^ a0));
}

__device__ void InvMixColumns(state_t* state) {
    int lane = threadIdx.x & 3;
    uchar4* stateVec = reinterpret_cast<uchar4*>(*state);
    uchar4 col = stateVec[lane];

    uint8_t s = mul0e(col.x) ^ mul0b(col.y) ^ mul0d(col.z) ^ mul09(col.w);
    uint8_t t = mul09(col.x) ^ mul0e(col.y) ^ mul0b(col.z) ^ mul0d(col.w);
    uint8_t u = mul0d(col.x) ^ mul09(col.y) ^ mul0e(col.z) ^ mul0b(col.w);
    uint8_t v = mul0b(col.x) ^ mul0d(col.y) ^ mul09(col.z) ^ mul0e(col.w);

    stateVec[lane] = make_uchar4(s, t, u, v);
}