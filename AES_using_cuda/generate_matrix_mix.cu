#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <vector>
#include <mma.h>

__constant__ half mix_matrix[32 * 32];

void init_mix_matrix() {
    std::vector<half> h_mix(32 * 32);

    for (int col = 0; col < 32; ++col) {
        uint32_t input_col = 1u << col;

        // AES MixColumns step
        auto xtime = [](uint8_t x) { return (uint8_t)((x << 1) ^ ((x >> 7) & 0x1b)); };

        uint8_t a0 = input_col & 0xff;
        uint8_t a1 = (input_col >> 8) & 0xff;
        uint8_t a2 = (input_col >> 16) & 0xff;
        uint8_t a3 = (input_col >> 24) & 0xff;
        uint8_t t = a0 ^ a1 ^ a2 ^ a3;
        uint8_t x0 = xtime(a0 ^ a1);
        uint8_t x1 = xtime(a1 ^ a2);
        uint8_t x2 = xtime(a2 ^ a3);
        uint8_t x3 = xtime(a3 ^ a0);
        uint32_t out_col = (x3 ^ t ^ a3) << 24 |
            (x2 ^ t ^ a2) << 16 |
            (x1 ^ t ^ a1) << 8 |
            (x0 ^ t ^ a0);

        for (int row = 0; row < 32; ++row) {
            int bit = (out_col >> row) & 1;
            h_mix[row * 32 + col] = __float2half((float)bit);
        }
    }

    cudaMemcpyToSymbol(mix_matrix, h_mix.data(), sizeof(half) * 32 * 32);
}