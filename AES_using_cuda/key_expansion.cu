#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include <openssl/evp.h>
#include <openssl/sha.h>
#include <iostream>
#include "aes_header.cuh"
// #include <cstring>
#include <vector>

// #define AES_KEY_SIZE 32 // 256-bit key size in bytes
// #define AES_EXPANDED_KEY_SIZE 240 // 240 bytes for AES-256 expanded key
// #define Nb 4 // Number of columns (32-bit words) comprising the state
// #define Nk 8 // Number of 32-bit words comprising the key
// #define Nr 14 // Number of rounds in AES-256

__device__ uint32_t RotWord(uint32_t word) {
    return (word << 8) | (word >> 24);
}

__device__ uint32_t SubWord(uint32_t word) {
    return (sbox[word & 0xFF]) |
        (sbox[(word >> 8) & 0xFF] << 8) |
        (sbox[(word >> 16) & 0xFF] << 16) |
        (sbox[(word >> 24) & 0xFF] << 24);
}

// Kernel for expanding the AES key
__global__ void keyExpansionKernel(uint8_t *expandedKey, const uint8_t* originalKey) {
    uint32_t temp;
    uint32_t* expKey = reinterpret_cast<uint32_t*>(expandedKey);
    const uint32_t* origKey = reinterpret_cast<const uint32_t*>(originalKey);

    const int idx = threadIdx.x;

    // Copy original key to expanded key
    if (idx < Nk) {
        expKey[idx] = origKey[idx];
    }
    __syncthreads();

    for (int i = Nk + idx; i < Nb * (Nr + 1); i += blockDim.x) {
        temp = expKey[i - 1];
        if (i % Nk == 0) {
            temp = SubWord(RotWord(temp)) ^ Rcon[(i / Nk) - 1];
        }
        else if (Nk > 6 && i % Nk == 4) {
            temp = SubWord(temp);
        }
        expKey[i] = expKey[i - Nk] ^ temp;
        __syncthreads();
    }
}

// Convert String to AES-compatible key
void convertStringToAESKey(const std::string& keyString, uint8_t* keyArray) {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, keyString.data(), keyString.size());
    SHA256_Final(keyArray, &sha256);
}



