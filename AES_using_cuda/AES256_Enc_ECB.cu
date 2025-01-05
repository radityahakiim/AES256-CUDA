#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <stdexcept>
//#include <fstream>

// Encryption
__global__ void AESEncryptKernel(state_t* states, uint8_t* RoundKey, size_t numBlocks) {
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
__global__ void AESDecryptKernel(state_t* states, uint8_t* RoundKey, size_t numBlocks) {
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
    delete[] originalKey;
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

    FILE* file_out = fopen(outputFile.c_str(), "wb");
    if (!file_out) {
        fclose(file_in);
        std::cerr << "Failed to open output file: " << outputFile << std::endl;
        return;
    }

    // Declare buffer data and size
    size_t bufferSize = 4096 * AES_BLOCK_SIZE;
    uint8_t* buffer = new uint8_t[bufferSize];
    uint8_t* d_buffer; // Device buffer
    size_t bytesRead;
    uint8_t* h_buffer = new uint8_t[bufferSize];

    while ((bytesRead = fread(buffer, 1, bufferSize, file_in)) > 0) {
        if (!isDecryption && feof(file_in)) {
            uint8_t padSize = AES_BLOCK_SIZE - (bytesRead % AES_BLOCK_SIZE);
            // ANSI X9.23
            if (padSize < AES_BLOCK_SIZE) {
                for (uint32_t i = 0; i < padSize; ++i) {
                    buffer[bytesRead + i] = (i == padSize - 1) ? padSize : 0;
                }
                bytesRead += padSize;
                std::cout << padSize << " bytes of padding added, new size: " << bytesRead << std::endl;
            }
        }
        // Grid and threads per block section
        size_t blockNum = bytesRead / AES_BLOCK_SIZE;
        size_t threadPblk = blockNum / num_sm;
        size_t maxThreads = static_cast<size_t>(prop.maxThreadsPerBlock);
        if (blockNum % num_sm > 0) threadPblk++;
        if (threadPblk > maxThreads) {
            threadPblk = maxThreads;
            num_sm = static_cast<int>(blockNum) / 1024;
            if (blockNum % 1024 > 0) {
                num_sm++;
            }
        }
        // blockNum = std::min(blockNum, (static_cast<size_t>(num_sm) * 8)); // Adjust based on profile
        std::cout << "Launching kernel with " << blockNum << " blocks, " << threadPblk << " threads per block\n";

        dim3 threadsPerBlock(static_cast<unsigned int>(threadPblk));
        dim3 blocksPerGrid(static_cast<unsigned int>(blockNum));

        // Device buffer allocation
        if (cudaMalloc(&d_buffer, bufferSize) != cudaSuccess) {
            std::cerr << "Failed to allocate device memory.\n";
            delete[] buffer;
            return;
        }

        if (cudaMemcpy(d_buffer, buffer, bytesRead, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "Failed to copy input to device memory.\n";
            delete[] buffer;
            cudaFree(d_buffer);
            return;
        }

        if (!isDecryption) {
            AESEncryptKernel << <blocksPerGrid, threadsPerBlock >> > ((state_t*)d_buffer, d_roundKey, blockNum);
        }
        else {
            AESDecryptKernel << <blocksPerGrid, threadsPerBlock >> > ((state_t*)d_buffer, d_roundKey, blockNum);
        }
        if ((err = cudaGetLastError()) != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err);
            delete[] buffer;
            cudaFree(d_buffer);
            return;
        }
        err = cudaDeviceSynchronize(); // Sychronize the kernel
        if (cudaGetLastError() != cudaSuccess) {
            std::cerr << "Kernel exec failed :" << cudaGetErrorString(err);
            delete[] buffer;
            cudaFree(d_buffer);
            return;
        }

        std::cout << "Kernel execution completed\n";

        // Copy back to host
        err = cudaMemcpy(h_buffer, d_buffer, bytesRead, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy result from device to host memory : " << cudaGetErrorString(err);
            delete[] buffer;
            delete[] h_buffer;
            cudaFree(d_buffer);
            cudaFree(d_roundKey);
            return;
        }

        // Remove the padding
        if (isDecryption && feof(file_in)) {
            uint32_t padding = h_buffer[bytesRead - 1];
            if (padding <= AES_BLOCK_SIZE) {
                bytesRead -= padding;
                std::cout << padding << " bytes of padding removed, new size: " << bytesRead << std::endl;
            }
            else {
                std::cerr << "Invalid padding detected post decryption. Padding byte: " << padding;
                delete[] buffer;
                delete[] h_buffer;
                cudaFree(d_buffer);
                cudaFree(d_roundKey);
                fclose(file_in);
                fclose(file_out);
                return;
            }
        }

        if (fwrite(h_buffer, 1, bytesRead, file_out) != bytesRead) {
            std::cerr << "Error writing decrypted data to file.\n";
        }
        cudaFree(d_buffer);
    }

    // Cleanup
    fclose(file_in);
    fclose(file_out);
    delete[] buffer;
    delete[] h_buffer;
    cudaFree(d_roundKey);

    std::cout << "Process completed\n";
}