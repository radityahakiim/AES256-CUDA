#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

__constant__ uint8_t d_expKey[AES_EXPANDED_KEY_SIZE];

__global__ void AESEncryptKernel(const uint8_t* plaintext, uint8_t* ciphertext, int numBlocks) {
	uint8_t state[AES_BLOCK_SIZE];
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	// Bound checking
	if (idx < numBlocks) {
		for (int i = 0; i < AES_BLOCK_SIZE; i++) {
			state[i] = plaintext[(idx * AES_BLOCK_SIZE) + i];
		}
	}
	else {
		return;
	}

	// Initial rounds
	AddRoundKey(state, d_expKey, AES_BLOCK_SIZE);

	// 13 Rounds for AES-256
	for (int round = 1; round < Nr; ++round) {
		SubBytes(state);
		ShiftRows(state);
		MixColumns(state);
		AddRoundKey(state, d_expKey + round * AES_BLOCK_SIZE, AES_BLOCK_SIZE);
	}
	// Final round
	SubBytes(state);
	ShiftRows(state);
	AddRoundKey(state, d_expKey + Nr * AES_BLOCK_SIZE, AES_BLOCK_SIZE);

	// Write the encrypted block to ciphertext
	for (int i = 0; i < AES_BLOCK_SIZE; ++i) {
		ciphertext[idx * AES_BLOCK_SIZE + i] = state[i];
	}
}

void h_AESEncryptECB(const uint8_t* plaintext, const uint8_t* key, uint8_t* ciphertext, size_t plaintextLength) {
	uint8_t* d_plaintext;
	uint8_t* d_ciphertext;
	// uint8_t* d_roundKeys;
	int numBlocks = (plaintextLength + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
	size_t plaintextSize = static_cast<size_t>(numBlocks) * AES_BLOCK_SIZE;

	cudaMalloc(&d_plaintext, plaintextSize);
	cudaMalloc(&d_ciphertext, plaintextSize);
	// cudaMalloc(&d_roundKeys, AES_EXPANDED_KEY_SIZE);

	// commit key expansion
	uint8_t* h_roundKeys = new uint8_t[AES_EXPANDED_KEY_SIZE];
	keyExpansion(h_roundKeys, key);

	// copy expanded key and sbox to constant data
	cudaMemcpyToSymbol(d_expKey, h_roundKeys, AES_EXPANDED_KEY_SIZE * sizeof(uint8_t)); // memcpy roundkeys or expanded keys to constant data
	cudaMemcpyToSymbol(d_sb, sbox, AES_KEY_SIZE * Nk * sizeof(uint8_t)); // memcpy sbox to constant data

	// copy plaintext  to gpu
	cudaMemcpy(d_plaintext, plaintext, plaintextSize, cudaMemcpyHostToDevice);

	// calculate threads/block and blocks/grid
	const int threadsPerBlock = AES_KEY_SIZE * Nk;
	const int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;

	// Encrypt
	AESEncryptKernel << <blocksPerGrid, threadsPerBlock >> > (d_plaintext, d_ciphertext, numBlocks);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
	}

	cudaMemcpy(ciphertext, d_ciphertext, plaintextSize, cudaMemcpyDeviceToHost);

	cudaFree(d_plaintext);
	cudaFree(d_ciphertext);
	delete[] h_roundKeys;
	cudaDeviceReset();
}