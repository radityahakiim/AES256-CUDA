#ifndef AES_HEADER_CUH
#define AES_HEADER_CUH

#include "cuda_runtime.h"
#include <cstdint>
#include <string>
#include <vector>

// Constants for AES-256
#define Nb 4 // Number of columns (32-bit words) comprising the state
#define Nk 8 // Number of 32-bit words comprising the key
#define Nr 14 // Number of rounds in AES-256
#define AES_BLOCK_SIZE (Nb * 4) // Define 128-bit for block size
#define AES_KEY_SIZE (Nb * Nk) // 256-bit key size in bytes
#define AES_EXPANDED_KEY_SIZE (Nb * (Nr + 1) * 4) // 240 bytes for AES-256 expanded key
#define PBKDF2_ITERATIONS 1000

extern uint8_t sbox[256];

// Define rcon
extern uint8_t Rcon[15];

// constant declaration for device uses
__constant__ extern uint8_t d_sb[256];
// __constant__ extern int d_Nb;
// __constant__ extern int d_Nr;
// __constant__ extern int d_Nk;
__constant__ extern uint8_t d_expKey[AES_EXPANDED_KEY_SIZE];

// Kernel for expanding the AES key (declaration only)
void keyExpansion(uint8_t* expandedKey, const uint8_t* originalKey);

// Kernel for AddRoundKey (declaration only)
__device__ void AddRoundKey(uint8_t* state, const uint8_t* roundKey, int numBlocks);

// Kernel for SubBytes (declaration only)
__device__ void SubBytes(uint8_t* state);

// Kernel for ShiftRows (declaration only)
__device__ void ShiftRows(uint8_t* state);

__device__ void MixColumns(uint8_t* state);

// Host function for converting a string to an AES-compatible key
void convertStringToAESKey(const std::string& keyString, uint8_t* keyArray);

//std::vector<uint8_t> preparePlaintext(const std::string& input);
void printHex(const std::vector<uint8_t>& data);

__global__ void XORWithKeyKernel(uint8_t* data, const uint8_t* key, int dataSize, int keySize);

// __global__ void AES256ECBKernel(uint8_t* block, const uint8_t* expandedKey);

__global__ void AESEncryptKernel(const uint8_t* plaintext, uint8_t* ciphertext, int numBlocks);
void h_AESEncryptECB(const uint8_t* plaintext, const uint8_t* key, uint8_t* ciphertext, size_t plaintextLength);

#endif // AES_CUH
