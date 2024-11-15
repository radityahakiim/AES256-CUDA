#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

__device__ uint8_t galois_mul2(uint8_t x) {
    return (x << 1) ^ ((x & 0x80) ? 0x1B : 0x00);
}

__device__ uint8_t galois_mul3(uint8_t x) {
    return galois_mul2(x) ^ x;
}

__device__ void MixColumns(uint8_t* state) {
    for (int i = 0; i < 4; i++){
        uint8_t s0 = state[i];
        uint8_t s1 = state[4 + i];
        uint8_t s2 = state[8 + i];
        uint8_t s3 = state[12 + i];

        state[i]      = galois_mul2(s0) ^ galois_mul3(s1) ^ s2 ^ s3;
        state[4 + i]  = s0 ^ galois_mul2(s1) ^ galois_mul3(s2) ^ s3;
        state[8 + i]  = s0^s1 ^ galois_mul2(s2) ^ galois_mul3(s3); 
        state[12 + i] = galois_mul3(s0) ^ s1 ^ s2 ^ galois_mul2(s3);
    }
}