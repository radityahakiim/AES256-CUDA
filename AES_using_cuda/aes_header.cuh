#ifndef AES_HEADER_CUH
#define AES_HEADER_CUH

#include "cuda_runtime.h"
#include <cstdint>
#include <string>
#include <mma.h>

// Constants for AES-256
#define Nb 4 // Number of columns (32-bit words) comprising the state
#define Nk 8 // Number of 32-bit words comprising the key
#define Nr 14 // Number of rounds in AES-256
#define AES_BLOCK_SIZE 16 // Define 128-bit for block size
#define AES_KEY_SIZE (Nb * Nk) // 256-bit key size in bytes
#define AES_EXPANDED_KEY_SIZE ((Nr + 1) * Nb * 4) // 240 bytes for AES-256 expanded key
#define NUM_STREAMS 8
//#define PBKDF2_ITERATIONS 1000

// extern uint8_t sbox[256];
// extern uint8_t inv_sbox[256];
// void SBoxInit(bool isDecryption);
uint8_t getSBoxValue(uint8_t num);
// __device__ uint8_t getSBoxValueDevice(uint8_t num);
// __device__ uint8_t getSBoxInvertDevice(uint8_t num);
typedef __align__(16) uint8_t state_t[4][4];

// Define rcon
extern uint8_t Rcon[8];

// constant declaration for device uses
// __constant__ extern int d_Nb;
// __constant__ extern int d_Nr;
// __constant__ extern int d_Nk;
extern __constant__ uint8_t d_sb[256];
extern __constant__ uint8_t d_inv_sb[256];

// For mix matrix
extern __constant__ half mix_matrix[32*32];
void init_mix_matrix();

// Kernel for expanding the AES key (declaration only)
void keyExpansion(uint32_t* expandedKey, const uint8_t* originalKey);

// Kernel for AddRoundKey (declaration only)
__device__ void AddRoundKey(state_t* state, uint8_t round, const uint32_t* roundKey);

// Kernel for SubBytes (declaration only)
__device__ void SubBytes(state_t* state);
__device__ void InvSubBytes(state_t* state); // Invert func

// Kernel for ShiftRows (declaration only)
__device__ void ShiftRows(state_t* state);
__device__ void InvShiftRows(state_t* state); // Invert func

// Kernel for MixColumns (declaration only
__device__ void MixColumns(state_t* state);
__device__ void InvMixColumns(state_t* state); // Invert func

// Host function for converting a string to an AES-compatible key
void convertStringToAESKey(const std::string& keyString, uint8_t* keyArray);

//std::vector<uint8_t> preparePlaintext(const std::string& input);
// void printHex(const std::vector<uint8_t>& data);

// __global__ void XORWithKeyKernel(uint8_t* data, const uint8_t* key, int dataSize, int keySize);

// __global__ void AESEncDecKernel(const uint8_t* plaintext, uint8_t* ciphertext, int numBlocks, int choice);
void h_AESEncDecECB(std::string inputFile, const std::string key, std::string output, bool isDecryption);
void h_AESEncDecCTR(std::string inputFile, const std::string key, std::string outputFile, bool isDecryption, uint64_t providedNonce = 0);

#endif // AES_CUH
