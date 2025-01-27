#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

static __device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

static __device__ __forceinline__ uint8_t Multiply(uint8_t x, uint8_t y) {
    uint8_t xt1 = xtime(x);
    uint8_t xt2 = xtime(xt1);
    uint8_t xt3 = xtime(xt2);
    uint8_t xt4 = xtime(xt3);

    return (((y & 1) * x) ^
        ((y >> 1 & 1) * xt1) ^
        ((y >> 2 & 1) * xt2) ^
        ((y >> 3 & 1) * xt3) ^
        ((y >> 4 & 1) * xt4));
}

__device__ void MixColumns(state_t* state) {
    int lane = threadIdx.x & 3;

    uint8_t a0 = (*state)[0][lane];
    uint8_t a1 = (*state)[1][lane];
    uint8_t a2 = (*state)[2][lane];
    uint8_t a3 = (*state)[3][lane];

    // Compute the MixColumns transformation using xtime and XORs
    uint8_t t = a0 ^ a1 ^ a2 ^ a3;
    uint8_t tmp0 = xtime(a0 ^ a1) ^ t ^ a0;
    uint8_t tmp1 = xtime(a1 ^ a2) ^ t ^ a1;
    uint8_t tmp2 = xtime(a2 ^ a3) ^ t ^ a2;
    uint8_t tmp3 = xtime(a3 ^ a0) ^ t ^ a3;

    // Synchronize the warp and shuffle results
    (*state)[0][lane] = tmp0;
    (*state)[1][lane] = tmp1;
    (*state)[2][lane] = tmp2;
    (*state)[3][lane] = tmp3;
}

__device__ void InvMixColumns(state_t* state) {
    int lane = threadIdx.x & 3;
    uint8_t a, b, c, d;
    a = (*state)[0][lane];
    b = (*state)[1][lane];
    c = (*state)[2][lane];
    d = (*state)[3][lane];

    (*state)[0][lane] = Multiply(a, 0x0e) ^ Multiply(b, 0x0b) ^ Multiply(c, 0x0d) ^ Multiply(d, 0x09);
    (*state)[1][lane] = Multiply(a, 0x09) ^ Multiply(b, 0x0e) ^ Multiply(c, 0x0b) ^ Multiply(d, 0x0d);
    (*state)[2][lane] = Multiply(a, 0x0d) ^ Multiply(b, 0x09) ^ Multiply(c, 0x0e) ^ Multiply(d, 0x0b);
    (*state)[3][lane] = Multiply(a, 0x0b) ^ Multiply(b, 0x0d) ^ Multiply(c, 0x09) ^ Multiply(d, 0x0e);
}