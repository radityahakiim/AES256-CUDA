#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"

static __device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return ((x << 1) ^ (-(x >> 7) & 1) & 0x1b);
}

static __device__ __forceinline__ uint8_t Multiply(uint8_t x, uint8_t y) {
    uint8_t xt1 = xtime(x);
    uint8_t xt2 = xtime(xt1);
    uint8_t xt3 = xtime(xt2);
    uint8_t xt4 = xtime(xt3);

    return ((-(y & 1) & x) ^
        (-(y >> 1 & 1) & xt1) ^
        (-(y >> 2 & 1) & xt2) ^
        (-(y >> 3 & 1) & xt3) ^
        (-(y >> 4 & 1) & xt4));
}

__device__ void MixColumns(state_t* state) {
    int lane = threadIdx.x & 3;

    uint8_t a0 = (*state)[0][lane];
    uint8_t a1 = (*state)[1][lane];
    uint8_t a2 = (*state)[2][lane];
    uint8_t a3 = (*state)[3][lane];

    // Compute the MixColumns transformation using xtime and XORs
    uint8_t t = a0 ^ a1 ^ a2 ^ a3;
    uint8_t u = xtime(a0 ^ a1);
    uint8_t v = xtime(a1 ^ a2);
    uint8_t w = xtime(a2 ^ a3);
    uint8_t x = xtime(a3 ^ a0);

    (*state)[0][lane] = u ^ t ^ a0;
    (*state)[1][lane] = v ^ t ^ a1;
    (*state)[2][lane] = w ^ t ^ a2;
    (*state)[3][lane] = x ^ t ^ a3;
}

__device__ void InvMixColumns(state_t* state) {
    int lane = threadIdx.x & 3;

    uint8_t a = (*state)[0][lane];
    uint8_t b = (*state)[1][lane];
    uint8_t c = (*state)[2][lane];
    uint8_t d = (*state)[3][lane];

    uint8_t s = Multiply(a, 0x0e) ^ Multiply(b, 0x0b) ^ Multiply(c, 0x0d) ^ Multiply(d, 0x09);
    uint8_t t = Multiply(a, 0x09) ^ Multiply(b, 0x0e) ^ Multiply(c, 0x0b) ^ Multiply(d, 0x0d);
    uint8_t u = Multiply(a, 0x0d) ^ Multiply(b, 0x09) ^ Multiply(c, 0x0e) ^ Multiply(d, 0x0b);
    uint8_t v = Multiply(a, 0x0b) ^ Multiply(b, 0x0d) ^ Multiply(c, 0x09) ^ Multiply(d, 0x0e);

    (*state)[0][lane] = s;
    (*state)[1][lane] = t;
    (*state)[2][lane] = u;
    (*state)[3][lane] = v;
}