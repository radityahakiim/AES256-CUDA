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
    uint8_t tmp, tm, t;
    #pragma unroll
    for (uint8_t i = 0; i < 4; i++){
            t   = (*state)[i][0];
            tmp = (*state)[i][0] ^ (*state)[i][1] ^ (*state)[i][2] ^ (*state)[i][3];
            tm  = (*state)[i][0] ^ (*state)[i][1]; tm = xtime(tm); (*state)[i][0] ^= tm ^ tmp;
            tm  = (*state)[i][1] ^ (*state)[i][2]; tm = xtime(tm); (*state)[i][1] ^= tm ^ tmp;
            tm  = (*state)[i][2] ^ (*state)[i][3]; tm = xtime(tm); (*state)[i][2] ^= tm ^ tmp;
            tm  = (*state)[i][3] ^ t             ; tm = xtime(tm); (*state)[i][3] ^= tm ^ tmp;
     }
}

__device__ void InvMixColumns(state_t* state) {
    uint8_t a, b, c, d;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a = (*state)[i][0];
        b = (*state)[i][1];
        c = (*state)[i][2];
        d = (*state)[i][3];

        (*state)[i][0] = Multiply(a, 0x0e) ^ Multiply(b, 0x0b) ^ Multiply(c, 0x0d) ^ Multiply(d, 0x09);
        (*state)[i][1] = Multiply(a, 0x09) ^ Multiply(b, 0x0e) ^ Multiply(c, 0x0b) ^ Multiply(d, 0x0d);
        (*state)[i][2] = Multiply(a, 0x0d) ^ Multiply(b, 0x09) ^ Multiply(c, 0x0e) ^ Multiply(d, 0x0b);
        (*state)[i][3] = Multiply(a, 0x0b) ^ Multiply(b, 0x0d) ^ Multiply(c, 0x09) ^ Multiply(d, 0x0e);
    }
}