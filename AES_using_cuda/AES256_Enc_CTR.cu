#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "aes_header.cuh"
#include <iostream>
#include <stdexcept>
#include <random>
//#include <fstream>

__constant__ uint32_t c_Rk_CTR[AES_EXPANDED_KEY_SIZE];

// Encryption
__global__ void AESCTRKernel(state_t* states, size_t numBlocks, uint64_t nonce, uint64_t counterStart) {
    extern __shared__ uint8_t shared_Mem[];
    state_t* sharedKeyStream = reinterpret_cast<state_t*>(shared_Mem);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBlocks) return;

    uint8_t state[4][4];
    state_t* keystream = &sharedKeyStream[threadIdx.x];
    uint32_t* src = reinterpret_cast<uint32_t*>(states[idx]);
    uint32_t* dst = reinterpret_cast<uint32_t*>(state);

#pragma unroll
    for (int i = 0; i < 4; i++) {
        dst[i] = src[i];
    }

    // Prepare counter block: nonce (8 bytes) || counter (8 bytes)
    uint64_t counter_val = counterStart + idx;
    uint8_t* counterBlock = reinterpret_cast<uint8_t*>(keystream);
    *reinterpret_cast<uint64_t*>(&counterBlock[0]) = nonce;
    *reinterpret_cast<uint64_t*>(&counterBlock[8]) = counter_val;

    // Initial rounds
    AddRoundKey(keystream, 0, c_Rk_CTR);
    // 13 Rounds for AES-256
#pragma unroll
    for (int round = 1; round < Nr; ++round) {
        SubBytes(keystream);
        ShiftRows(keystream);
        MixColumns(keystream);
        AddRoundKey(keystream, round, c_Rk_CTR);
    }
    // Final round
    SubBytes(keystream);
    ShiftRows(keystream);
    AddRoundKey(keystream, Nr, c_Rk_CTR);

    // XOR state with keystream
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            state[i][j] ^= (*keystream)[i][j];
        }
    }

    // Write back to global memory
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            states[idx][i][j] = state[i][j];
        }
    }
}

uint64_t generateNonce() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
    return dis(gen);
}

void h_AESEncDecCTR(std::string inputFile, const std::string key, std::string outputFile, bool isDecryption, uint64_t providedNonce) {
    std::cout << "Starting CTR " << (isDecryption ? "Decryption" : "Encryption") << " process with input: " << inputFile << ", output: " << outputFile << std::endl;

    // Device preparation
    cudaError_t err;
    cudaSetDevice(0); // Using main GPU
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::cerr << "Failed to get device properties.\n";
        return;
    }
    // constexpr size_t PADDED_STATE_SIZE = sizeof(state_t) + (16 - (sizeof(state_t) % 16)) % 16; // Padded size for shared memory
    int num_sm = prop.multiProcessorCount;
    std::cout << "Using a GPU with " << num_sm << " SMs\n";

    // Round Key section
    uint8_t* originalKey;
    cudaMallocHost(&originalKey, AES_KEY_SIZE);
    uint32_t* expandedKey;
    cudaMallocHost(&expandedKey, AES_EXPANDED_KEY_SIZE);

    // Key processing
    convertStringToAESKey(key, originalKey); // String key hash using SHA-256
    std::cout << "Key converted to binary\n";
    keyExpansion(expandedKey, originalKey);
    cudaFreeHost(originalKey);
    std::cout << "Key expanded\n";

    // Copy Round key to device
    // uint8_t* d_roundKey;
    // cudaMalloc(&d_roundKey, AES_EXPANDED_KEY_SIZE);
    cudaMemcpyToSymbol(c_Rk_CTR, expandedKey, AES_EXPANDED_KEY_SIZE);
    cudaFreeHost(expandedKey);
    std::cout << "Expanded key copied to constant\n";

    // S-box initialization
    SBoxInit(false);
    std::cout << "S-box initialized for CTR" << std::endl;

    // CUDA Events timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float totalTime = 0.0;
    int kernelExec = 0;

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

    // Handle nonce: generate for encryption, read for decryption
    uint64_t nonce;
    size_t inputOffset = 0;
    if (isDecryption) {
        if (fread(&nonce, sizeof(nonce), 1, file_in) != 1) {
            std::cerr << "Failed to read nonce from input file\n";
            fclose(file_in);
            fclose(file_out);
            return;
        }
        inputOffset = sizeof(nonce);
        std::cout << "Nonce: " << nonce << std::endl;
    }
    else {
        nonce = (providedNonce != 0) ? providedNonce : generateNonce();
        if (fwrite(&nonce, sizeof(nonce), 1, file_out) != 1) {
            std::cerr << "Failed to write nonce to output file\n";
            fclose(file_in);
            fclose(file_out);
            return;
        }
        std::cout << "Generated nonce: " << nonce << std::endl;
    }

    // Declare buffer data and size
    size_t bufferSize = (static_cast<size_t>(1024) * 1024);
    uint8_t* buffer;
    uint8_t* d_buffer; // Device buffer
    size_t bytesRead;
    // Host alloc
    cudaMallocHost(&buffer, bufferSize);
    // Device buffer allocation
    if (cudaMalloc(&d_buffer, ((bufferSize + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE) * sizeof(state_t)) != cudaSuccess) {
        cudaDeviceReset();
        return;
    }

    uint64_t globalCounter = 0;

    if (isDecryption) {
        fseek(file_in, inputOffset, SEEK_SET);
    }

    while ((bytesRead = fread(buffer, 1, bufferSize, file_in)) > 0) {
        // Block number setup
        size_t blockNum = (bytesRead + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
        // size_t alignedBytes = blockNum * AES_BLOCK_SIZE;
        // size_t bp_grid = (blockNum + threadPblk - 1) / threadPblk;

        // Convert buffer to state_t array
        state_t* hostStates;
        cudaMallocHost(&hostStates, blockNum * sizeof(state_t));
        for (size_t i = 0; i < blockNum; i++) {
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    size_t byteIdx = i * AES_BLOCK_SIZE + (r * 4 + c);
                    hostStates[i][r][c] = (byteIdx < bytesRead) ? buffer[byteIdx] : 0;
                }
            }
        }

        // Set up thread and grid
        size_t maxThreads = static_cast<size_t>(prop.maxThreadsPerBlock);
        size_t threadPblk = blockNum / num_sm;
        if (blockNum % num_sm > 0) threadPblk++;
        if (threadPblk > maxThreads) {
            threadPblk = maxThreads;
            num_sm = static_cast<int>(blockNum) / 1024;
            if (blockNum % 1024 > 0) {
                num_sm++;
            }
        }
        size_t shmemSize = sizeof(state_t) * threadPblk;
        // blockNum = std::min(blockNum, (static_cast<size_t>(num_sm) * 8)); // Adjust based on profile
        std::cout << "Launching kernel with " << num_sm << " blocks, " << threadPblk << " threads per block\n";

        dim3 threadsPerBlock(static_cast<unsigned int>(threadPblk));
        dim3 blocksPerGrid(static_cast<unsigned int>(num_sm));

        if (cudaMemcpy(d_buffer, hostStates, blockNum * sizeof(state_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "Failed to copy input to device memory.\n";
            cudaDeviceReset();
            return;
        }

        cudaEventRecord(start);

        AESCTRKernel << <blocksPerGrid, threadsPerBlock, shmemSize >> > ((state_t*)d_buffer, blockNum, nonce, globalCounter);

        if ((err = cudaGetLastError()) != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err);
            cudaDeviceReset();
            return;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Kernel execution completed in " << milliseconds << " ms\n";

        // Accumulate the time and count the execution
        totalTime += milliseconds;
        kernelExec++;

        // Copy back to host
        err = cudaMemcpy(hostStates, d_buffer, blockNum * sizeof(state_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy result from device to host memory : " << cudaGetErrorString(err);
            cudaDeviceReset();
            return;
        }

        for (size_t i = 0; i < blockNum; i++) {
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    size_t byteIdx = i * AES_BLOCK_SIZE + (r * 4 + c);
                    if (byteIdx < bytesRead) {
                        buffer[byteIdx] = hostStates[i][r][c];
                    }
                }
            }
        }
        cudaFreeHost(hostStates);

        if (fwrite(buffer, 1, bytesRead, file_out) != bytesRead) {
            std::cerr << "Error writing decrypted data to file.\n";
            cudaDeviceReset();
            return;
        }
        globalCounter += blockNum;
    }

    // Cleanup
    fclose(file_in);
    fclose(file_out);
    cudaFreeHost(buffer);
    cudaFree(d_buffer);
    // cudaFree(d_roundKey);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calculate and print average time
    if (kernelExec > 0) {
        double averageTime = totalTime / kernelExec;
        std::cout << "Total time: " << totalTime << " ms\n";
        std::cout << "Times kernel executed: " << kernelExec << " times\n";
        std::cout << "Average kernel execution time: " << averageTime << " ms\n";
    }
    else {
        std::cout << "No kernels were executed.\n";
    }

    std::cout << "Process completed\n";
}