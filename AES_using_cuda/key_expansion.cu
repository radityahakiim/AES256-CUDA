#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <openssl/sha.h>
#include <iostream>
#include "aes_header.cuh"
// #include <cstring>
#include <vector>

// Define rcon
extern uint8_t Rcon[8] = { 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40 };

// Function for expanding the AES key
void keyExpansion(uint32_t* expandedKey, const uint8_t* originalKey) {
    uint32_t temp;
    unsigned i;

    // Copy original key to expanded key
    for (i = 0; i < Nk; i++) {
        expandedKey[i] = (originalKey[(i * 4) + 0] << 24) |
            (originalKey[(i * 4) + 1] << 16) |
            (originalKey[(i * 4) + 2] << 8) |
            originalKey[(i * 4) + 3];
    }

    // Key expansion loop
    for (i = Nk; i < Nb * (Nr + 1); i++) {
        temp = expandedKey[i - 1];

        if (i % Nk == 0) {
            // Rotate
            temp = (temp << 8) | ((temp >> 24) & 0xFF);

            // Substitute with S-box values each byte
            temp = (getSBoxValue((temp >> 24) & 0xFF) << 24) |
                (getSBoxValue((temp >> 16) & 0xFF) << 16) |
                (getSBoxValue((temp >> 8) & 0xFF) << 8) |
                getSBoxValue(temp & 0xFF);

            // Rcon XOR
            temp = temp ^ (Rcon[i / Nk] << 24);
        }
        else if (Nk > 6 && i % Nk == 4)
        {
            // For AES-256, apply S-box to each byte
            temp = (getSBoxValue((temp >> 24) & 0xFF) << 24) |
                (getSBoxValue((temp >> 16) & 0xFF) << 16) |
                (getSBoxValue((temp >> 8) & 0xFF) << 8) |
                getSBoxValue(temp & 0xFF);
        }

        // XOR with the word Nk positions back
        expandedKey[i] = expandedKey[i - Nk] ^ temp;
    }
}

// Convert String to AES-compatible key
void convertStringToAESKey(const std::string& keyString, uint8_t* keyArray) {
    SHA256(reinterpret_cast<const uint8_t*>(keyString.data()), keyString.size(), keyArray);
}