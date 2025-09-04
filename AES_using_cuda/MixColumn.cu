#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

static __device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return ((x << 1) ^ ((x >> 7) & 0x1b));
}

__constant__ uint32_t M[32] = {
    0x01018180u, 0x02028381u, 0x04040602u, 0x08088C84u,
    0x10109888u, 0x20203010u, 0x40406020u, 0x8080C040u,
    0x01818001u, 0x02838102u, 0x04060204u, 0x088C8408u,
    0x10988810u, 0x20301020u, 0x40602040u, 0x80C04080u,
    0x81800101u, 0x83810202u, 0x06020404u, 0x8C840808u,
    0x98881010u, 0x30102020u, 0x60204040u, 0xC0408080u,
    0x80010181u, 0x81020283u, 0x02040406u, 0x8408088Cu,
    0x88101098u, 0x10202030u, 0x20404060u, 0x408080C0u
};

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
    const unsigned mask_all = 0xffffffffu;
    unsigned lane = threadIdx.x & 31u;

    // Load one row of MixColumns matrix into each lane
    uint32_t M_row = 0;
    if (lane < 32u) {
        M_row = M[lane];
    }

    // Process column 0
    uint32_t input_bits0 = 0u;
    if (lane == 0u) {
        input_bits0 = (uint32_t)(*state)[0][0] | ((uint32_t)(*state)[0][1] << 8) | ((uint32_t)(*state)[0][2] << 16) | ((uint32_t)(*state)[0][3] << 24);
    }
    input_bits0 = __shfl_sync(mask_all, input_bits0, 0);
    unsigned p0 = (__popc(M_row & input_bits0) & 1u);
    uint32_t new_bits0 = __ballot_sync(mask_all, p0 != 0);
    if (lane == 0u) {
        (*state)[0][0] = new_bits0 & 0xFFu;
        (*state)[0][1] = (new_bits0 >> 8) & 0xFFu;
        (*state)[0][2] = (new_bits0 >> 16) & 0xFFu;
        (*state)[0][3] = (new_bits0 >> 24) & 0xFFu;
    }

    // Process column 1
    uint32_t input_bits1 = 0u;
    if (lane == 0u) {
        input_bits1 = (uint32_t)(*state)[1][0] | ((uint32_t)(*state)[1][1] << 8) | ((uint32_t)(*state)[1][2] << 16) | ((uint32_t)(*state)[1][3] << 24);
    }
    input_bits1 = __shfl_sync(mask_all, input_bits1, 0);
    unsigned p1 = (__popc(M_row & input_bits1) & 1u);
    uint32_t new_bits1 = __ballot_sync(mask_all, p1 != 0);
    if (lane == 0u) {
        (*state)[1][0] = new_bits1 & 0xFFu;
        (*state)[1][1] = (new_bits1 >> 8) & 0xFFu;
        (*state)[1][2] = (new_bits1 >> 16) & 0xFFu;
        (*state)[1][3] = (new_bits1 >> 24) & 0xFFu;
    }

    // Process column 2
    uint32_t input_bits2 = 0u;
    if (lane == 0u) {
        input_bits2 = (uint32_t)(*state)[2][0] | ((uint32_t)(*state)[2][1] << 8) | ((uint32_t)(*state)[2][2] << 16) | ((uint32_t)(*state)[2][3] << 24);
    }
    input_bits2 = __shfl_sync(mask_all, input_bits2, 0);
    unsigned p2 = (__popc(M_row & input_bits2) & 1u);
    uint32_t new_bits2 = __ballot_sync(mask_all, p2 != 0);
    if (lane == 0u) {
        (*state)[2][0] = new_bits2 & 0xFFu;
        (*state)[2][1] = (new_bits2 >> 8) & 0xFFu;
        (*state)[2][2] = (new_bits2 >> 16) & 0xFFu;
        (*state)[2][3] = (new_bits2 >> 24) & 0xFFu;
    }

    // Process column 3
    uint32_t input_bits3 = 0u;
    if (lane == 0u) {
        input_bits3 = (uint32_t)(*state)[3][0] | ((uint32_t)(*state)[3][1] << 8) | ((uint32_t)(*state)[3][2] << 16) | ((uint32_t)(*state)[3][3] << 24);
    }
    input_bits3 = __shfl_sync(mask_all, input_bits3, 0);
    unsigned p3 = (__popc(M_row & input_bits3) & 1u);
    uint32_t new_bits3 = __ballot_sync(mask_all, p3 != 0);
    if (lane == 0u) {
        (*state)[3][0] = new_bits3 & 0xFFu;
        (*state)[3][1] = (new_bits3 >> 8) & 0xFFu;
        (*state)[3][2] = (new_bits3 >> 16) & 0xFFu;
        (*state)[3][3] = (new_bits3 >> 24) & 0xFFu;
    }
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