#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <stdexcept>
//#include <fstream>

__constant__ uint32_t c_Rk[AES_EXPANDED_KEY_SIZE];

// Encryption
__global__ void AESEncryptKernel(state_t* states, size_t numBlocks) {
    extern __shared__ uint8_t shared_Mem[];
    state_t* sharedState = reinterpret_cast<state_t*>(shared_Mem);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBlocks) return;

    uint32_t* gmem_ptr = reinterpret_cast<uint32_t*>(&states[idx]);
    uint32_t* shmem_ptr = reinterpret_cast<uint32_t*>(&sharedState[threadIdx.x]);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        shmem_ptr[i] = __ldg(&gmem_ptr[i]);
    }

    __syncthreads();

    state_t* state = &sharedState[threadIdx.x];
    // Initial rounds
    AddRoundKey(state, 0, c_Rk);
    // 13 Rounds for AES-256
#pragma unroll
    for (int round = 1; round < Nr; ++round) {
        SubBytes(state);
        ShiftRows(state);
        MixColumns(state);
        AddRoundKey(state, round, c_Rk);
    }
    // Final round
    SubBytes(state);
    ShiftRows(state);
    AddRoundKey(state, Nr, c_Rk);

    __syncthreads();

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        gmem_ptr[i] = shmem_ptr[i];
    }
}

// Decryption
__global__ void AESDecryptKernel(state_t* states, size_t numBlocks) {
    extern __shared__ uint8_t shared_Mem[];
    state_t* sharedState = reinterpret_cast<state_t*>(shared_Mem);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBlocks) return;

    uint32_t* gmem_ptr = reinterpret_cast<uint32_t*>(&states[idx]);
    uint32_t* shmem_ptr = reinterpret_cast<uint32_t*>(&sharedState[threadIdx.x]);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        shmem_ptr[i] = __ldg(&gmem_ptr[i]);
    }

    __syncthreads();
    state_t* state = &sharedState[threadIdx.x];

    // Initial rounds
    AddRoundKey(state, Nr, c_Rk);
    // 13 Rounds for AES-256 (decryption)
#pragma unroll
    for (int round = Nr - 1; round > 0; --round) {
        InvShiftRows(state);
        InvSubBytes(state);
        AddRoundKey(state, round, c_Rk);
        InvMixColumns(state);
    }
    // Final round
    InvShiftRows(state);
    InvSubBytes(state);
    AddRoundKey(state, 0, c_Rk);

    __syncthreads();

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        gmem_ptr[i] = shmem_ptr[i];
    }
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
    constexpr size_t PADDED_STATE_SIZE = sizeof(state_t) + 16 - (sizeof(state_t) % 16); // Padded size for shared memory
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
    cudaMemcpyToSymbol(c_Rk, expandedKey, AES_EXPANDED_KEY_SIZE);
    cudaFreeHost(expandedKey);
    std::cout << "Expanded key copied to constant\n";

    // S-box initialization
    SBoxInit(isDecryption);
    std::cout << "S-box initialized for " << (isDecryption ? "decryption" : "encryption") << std::endl;

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

    // Declare buffer data and size
    size_t bufferSize = (static_cast<size_t>(1024) * 1024);
    uint8_t* buffer;
    uint8_t* d_buffer; // Device buffer
    size_t bytesRead;
    // Host alloc
    cudaMallocHost(&buffer, bufferSize);
    // Device buffer allocation
    if (cudaMalloc(&d_buffer, bufferSize) != cudaSuccess) {
        cudaDeviceReset();
        return;
    }

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
        size_t maxThreads = static_cast<size_t>(prop.maxThreadsPerBlock);
        size_t blockNum = bytesRead / AES_BLOCK_SIZE;
        size_t threadPblk = blockNum / num_sm;
        // size_t bp_grid = (blockNum + threadPblk - 1) / threadPblk;
        if (blockNum % num_sm > 0) threadPblk++;
        if (threadPblk > maxThreads) {
            threadPblk = maxThreads;
            num_sm = static_cast<int>(blockNum) / 1024;
            if (blockNum % 1024 > 0) {
                num_sm++;
            }
        }
        // blockNum = std::min(blockNum, (static_cast<size_t>(num_sm) * 8)); // Adjust based on profile
        std::cout << "Launching kernel with " << num_sm << " blocks, " << threadPblk << " threads per block\n";

        dim3 threadsPerBlock(static_cast<unsigned int>(threadPblk));
        dim3 blocksPerGrid(static_cast<unsigned int>(num_sm));
        size_t shmemSize = PADDED_STATE_SIZE * threadPblk;

        if (cudaMemcpy(d_buffer, buffer, bytesRead, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "Failed to copy input to device memory.\n";
            cudaDeviceReset();
            return;
        }

        cudaEventRecord(start);

        if (!isDecryption) {
            AESEncryptKernel << <blocksPerGrid, threadsPerBlock, shmemSize >> > ((state_t*)d_buffer, blockNum);
        }
        else {
            AESDecryptKernel << <blocksPerGrid, threadsPerBlock, shmemSize >> > ((state_t*)d_buffer, blockNum);
        }
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
        err = cudaMemcpy(buffer, d_buffer, bytesRead, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy result from device to host memory : " << cudaGetErrorString(err);
            cudaDeviceReset();
            return;
        }

        // Remove the padding
        if (isDecryption && feof(file_in)) {
            uint32_t padding = buffer[bytesRead - 1];
            if (padding <= AES_BLOCK_SIZE) {
                bytesRead -= padding;
                std::cout << padding << " bytes of padding removed, new size: " << bytesRead << std::endl;
            }
            else {
                std::cerr << "Invalid padding detected post decryption. Padding byte: " << padding;
                cudaDeviceReset();
                return;
            }
        }

        if (fwrite(buffer, 1, bytesRead, file_out) != bytesRead) {
            std::cerr << "Error writing decrypted data to file.\n";
            cudaDeviceReset();
            return;
        }
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