#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <openssl/sha.h>
#include <openssl/rand.h>
#include <iostream>
#include "aes_header.cuh"
// #include <cstring>
#include <vector>

// Define rcon
extern uint8_t Rcon[8] = { 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40 };

// Function for expanding the AES key
void keyExpansion(uint8_t* expandedKey, const uint8_t* originalKey) {
    uint32_t temp[4];
    unsigned i, j, k;

    // Copy original key to expanded key
    for (i = 0; i < Nk; i++) {
        expandedKey[(i * 4) + 0] = originalKey[(i * 4) + 0];
        expandedKey[(i * 4) + 1] = originalKey[(i * 4) + 1];
        expandedKey[(i * 4) + 2] = originalKey[(i * 4) + 2];
        expandedKey[(i * 4) + 3] = originalKey[(i * 4) + 3];
    }

    // Key expansion loop
    for (i = Nk; i < Nb * (Nr + 1); i++) {
        k = (i - 1) * 4;
        temp[0] = expandedKey[k + 0];
        temp[1] = expandedKey[k + 1];
        temp[2] = expandedKey[k + 2];
        temp[3] = expandedKey[k + 3];

        if (i % Nk == 0) {
            const uint8_t tempb = temp[0];
            temp[0] = temp[1];
            temp[1] = temp[2];
            temp[2] = temp[3];
            temp[3] = tempb;

            temp[0] = getSBoxValue(temp[0]);
            temp[1] = getSBoxValue(temp[1]);
            temp[2] = getSBoxValue(temp[2]);
            temp[3] = getSBoxValue(temp[3]);

            temp[0] = temp[0] ^ Rcon[i / Nk];
        }
        if (i % Nk == 4) {
            temp[0] = getSBoxValue(temp[0]);
            temp[1] = getSBoxValue(temp[1]);
            temp[2] = getSBoxValue(temp[2]);
            temp[3] = getSBoxValue(temp[3]);
        }

        j = i * 4; k = (i - Nk) * 4;
        expandedKey[j + 0] = expandedKey[k + 0] ^ temp[0];
        expandedKey[j + 1] = expandedKey[k + 1] ^ temp[1];
        expandedKey[j + 2] = expandedKey[k + 2] ^ temp[2];
        expandedKey[j + 3] = expandedKey[k + 3] ^ temp[3];
    }
}

// Convert String to AES-compatible key
void convertStringToAESKey(const std::string& keyString, uint8_t* keyArray) {
    SHA256(reinterpret_cast<const uint8_t*>(keyString.data()), keyString.size(), keyArray);
}