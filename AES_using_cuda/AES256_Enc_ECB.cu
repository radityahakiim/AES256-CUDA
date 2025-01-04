#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <stdexcept>
//#include <fstream>

// Encryption
__global__ void AESEncryptKernel(state_t* states, uint8_t* RoundKey, int numBlocks) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numBlocks) return;
	state_t* state = &states[idx];
    
	// Initial rounds
	AddRoundKey(state, 0, RoundKey);
		// 13 Rounds for AES-256
		for (int round = 1; round < Nr; ++round) {
			SubBytes(state);
			ShiftRows(state);
			MixColumns(state);
			AddRoundKey(state, round, RoundKey);
		}
		// Final round
		SubBytes(state);
		ShiftRows(state);
		AddRoundKey(state, Nr, RoundKey);
}

// Decryption
__global__ void AESDecryptKernel(state_t* states, uint8_t* RoundKey, int numBlocks) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numBlocks) return;
	state_t* state = &states[idx];

	// Initial rounds
	AddRoundKey(state, Nr, RoundKey);
	// 13 Rounds for AES-256 (decryption)
	for (int round = Nr - 1; round > 0; --round) {
		InvShiftRows(state);
		InvSubBytes(state);
		AddRoundKey(state, round, RoundKey);
		InvMixColumns(state);
	}
	// Final round
	InvShiftRows(state);
	InvSubBytes(state);
	AddRoundKey(state, 0, RoundKey);
}

void h_AESEncDecECB(std::string inputFile, const std::string key, std::string outputFile, bool isDecryption) {
    std::cout << "Starting " << (isDecryption ? "Decryption" : "Encryption") << " process with input: " << inputFile << ", output: " << outputFile << std::endl;

    // Device preparation
    cudaError_t err;
    cudaSetDevice(0); // Using main GPU
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::cerr << "Failed to get device properties.\n";
        return;
    }
    int num_sm = prop.multiProcessorCount;
    std::cout << "Using a GPU with " << num_sm << " SMs\n";

    // Round Key section
    uint8_t* originalKey = new uint8_t[AES_KEY_SIZE];
    uint8_t* expandedKey = new uint8_t[AES_EXPANDED_KEY_SIZE];

    // Key processing
    convertStringToAESKey(key, originalKey); // String key hash using SHA-256
    std::cout << "Key converted to binary\n";
    keyExpansion(expandedKey, originalKey);
    std::cout << "Key expanded\n";

    // Copy Round key to device
    uint8_t* d_roundKey;
    cudaMalloc(&d_roundKey, AES_EXPANDED_KEY_SIZE);
    cudaMemcpy(d_roundKey, expandedKey, AES_EXPANDED_KEY_SIZE, cudaMemcpyHostToDevice);
    delete[] expandedKey;
    std::cout << "Expanded key copied to device\n";

    // S-box initialization
    SBoxInit(isDecryption);
    std::cout << "S-box initialized for " << (isDecryption ? "decryption" : "encryption") << std::endl;

    // File processing 
    FILE* file_in = fopen(inputFile.c_str(), "rb");
    if (!file_in) {
        std::cerr << "Failed to open input file: " << inputFile << std::endl;
        return;
    }

    FILE* paddedFile = fopen(outputFile.c_str(), "wb+");
    if (!paddedFile) {
        fclose(file_in);
        std::cerr << "Failed to open output file: " << outputFile << std::endl;
        return;
    }

    // Copy input to output
    char buffer[4096];
    size_t bytesRead;
    while ((bytesRead = fread(buffer, 1, sizeof(buffer), file_in)) > 0) {
        if (fwrite(buffer, 1, bytesRead, paddedFile) != bytesRead) {
            std::cerr << "Error writing to output file." << std::endl;
            fclose(file_in);
            fclose(paddedFile);
            return;
        }
    }
    fclose(file_in);
    std::cout << "File copied to " << outputFile << std::endl;

    fseek(paddedFile, 0, SEEK_END);
    uint32_t size = ftell(paddedFile);
    std::cout << "File size after copy: " << size << " bytes\n";

    if (!isDecryption) {
        uint32_t len = AES_BLOCK_SIZE - (size % AES_BLOCK_SIZE);
        uint32_t* pad = new uint32_t[len];
        // ANSI X9.23 padding
        for (uint32_t i = 0; i < len - 1; i++) {
            pad[i] = 0x00;
        }
        pad[len - 1] = len;
        if (fwrite(pad, sizeof(uint32_t), len, paddedFile) != len) {
            std::cerr << "Error padding the file." << std::endl;
            delete[] pad;
            fclose(paddedFile);
            return;
        }
        delete[] pad;
        size += len;
        std::cout << len << " bytes of padding added, new size: " << size << std::endl;
    }

    rewind(paddedFile);
    uint8_t* input = new uint8_t[size];
    if (fread(input, 1, size, paddedFile) != size) {
        std::cerr << "Error reading padded file into memory." << std::endl;
        delete[] input;
        fclose(paddedFile);
        return;
    }
    fclose(paddedFile);
    std::cout << "Input data read into memory\n";

    // Grid and threads per block section
    int blockNum = (size + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
    int threadPblk = std::min(256, prop.maxThreadsPerBlock); // Good starting point, adjust based on performance
    blockNum = std::min(blockNum, prop.multiProcessorCount * 8); // Adjust based on profile
    std::cout << "Launching kernel with " << blockNum << " blocks, " << threadPblk << " threads per block\n";

    dim3 threadsPerBlock(threadPblk);
    dim3 blocksPerGrid(blockNum);

    // Device mem. allocation
    uint8_t* d_input;
    if (cudaMalloc(&d_input, size) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory.\n";
        delete[] input;
        return;
    }
    if (cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy input to device memory.\n";
        delete[] input;
        cudaFree(d_input);
        return;
    }

    if (!isDecryption) {
        AESEncryptKernel << <blocksPerGrid, threadsPerBlock >> > ((state_t*)d_input, d_roundKey, blockNum);
    }
    else {
        AESDecryptKernel << <blocksPerGrid, threadsPerBlock >> > ((state_t*)d_input, d_roundKey, blockNum);
    }
    err = cudaDeviceSynchronize(); // Sychronize the kernel
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Kernel launch failed :" << cudaGetErrorString(err);
        delete[] input;
        cudaFree(d_input);
        return;
    }

    std::cout << "Kernel execution completed\n";

    // Copy back to host
    uint8_t* h_result = new uint8_t[size];
    err = cudaMemcpy(h_result, d_input, size, cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
        std::cerr << "Failed to copy result from device to host memory : " << cudaGetErrorString(err);
        delete[] input;
        delete[] h_result;
        cudaFree(d_input);
        return;
    }

    // Remove the padding
        if (isDecryption) {
            uint32_t del = h_result[size - 1];
            if (del <= AES_BLOCK_SIZE) {
                size -= del;
                std::cout << del << " bytes of padding removed, new size: " << size << std::endl;
            }
            else {
                std::cerr << "Invalid padding detected post decryption. Padding byte: " << del;
                return;
            }
        }

    FILE* file_out = fopen(outputFile.c_str(), "wb");
    if (!file_out) {
        std::cerr << "Failed to open output file for writing result.\n";
        delete[] input;
        delete[] h_result;
        cudaFree(d_input);
        return;
    }
    if (fwrite(h_result, sizeof(uint8_t), size, file_out) != size) {
        std::cerr << "Error writing decrypted data to file.\n";
    }
    fclose(file_out);
    std::cout << "Result written to file\n";

    // Cleanup
    delete[] input;
    delete[] h_result;
    delete[] originalKey;
    cudaFree(d_input);

    std::cout << "Process completed\n";
}