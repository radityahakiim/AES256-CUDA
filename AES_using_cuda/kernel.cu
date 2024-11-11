#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

int main() {
	std::string textKey = "passwordrahasia1234567890";
	std::string plaintext = "hello";
	uint8_t originalKey[AES_KEY_SIZE];
	convertStringToAESKey(textKey, originalKey);

	uint8_t h_expandedKey[AES_EXPANDED_KEY_SIZE];
	uint8_t h_originalKey[AES_KEY_SIZE];
	uint8_t *d_originalKey, *d_expandedKey, *d_state;

	// allocate mem on device
	cudaMalloc((void**)&d_originalKey, AES_KEY_SIZE * sizeof(uint8_t));
	cudaMalloc((void**)&d_expandedKey, AES_EXPANDED_KEY_SIZE * sizeof(uint8_t));
	cudaMalloc((void**)&d_state, Nb * 4);

	// Copy original key to device
	cudaMemcpy(d_originalKey, originalKey, AES_KEY_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);

	// Launch kernel
	int threadsPerBlock = 8;
	keyExpansionKernel << < 1, threadsPerBlock >> > (d_expandedKey, d_originalKey);
	cudaDeviceSynchronize();

	uint8_t roundKey[16]; // Start of applying the AddRoundKey for round 0, assuming 16 bytes for one round key first
	// Copy expanded key back to host
	cudaMemcpy(h_originalKey, d_originalKey, AES_KEY_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_expandedKey, d_expandedKey, AES_EXPANDED_KEY_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	AddRoundKey<<<1, AES_BLOCK_SIZE>>>(d_state, roundKey);
	cudaDeviceSynchronize();

	// Memory cleanup
	cudaFree(d_originalKey);
	cudaFree(d_expandedKey);
	cudaFree(d_state);

	std::vector<uint8_t> plaintextBlocks = preparePlaintext(plaintext);
	std::cout << "Padded Plaintext in hex :" << std::endl;
	printHex(plaintextBlocks);

	std::cout << "Original Key: " << std::endl;
	for (int i = 0; i < AES_KEY_SIZE; i++) {
		printf("%02x ", h_originalKey[i]);
		if ((i + 1) % 16 == 0) std::cout << std::endl;
	}

	//Print the expanded key
	std::cout << "Expanded Key: " << std::endl;
	for (int i = 0; i < AES_EXPANDED_KEY_SIZE; i++) {
		printf("%02x ", h_expandedKey[i]);
	 	if ((i + 1) % 16 == 0) std::cout << std::endl;
	}
	return 0;
}
