#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <stdexcept>
//#include <fstream>

__constant__ uint32_t c_Rk[AES_EXPANDED_KEY_SIZE];

// Device encryption
__device__ void AESEncryptDevice(state_t* state) {
    // Initial round
    AddRoundKey(state, 0, c_Rk);

    // 13 Rounds for AES-256
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
}

// Device decryption
__device__ void AESDecryptDevice(state_t* state) {
    // Initial rounds
    AddRoundKey(state, Nr, c_Rk);
    // 13 Rounds for AES-256 (decryption)
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
}

// Encryption
__global__ void AESEncryptKernel(state_t* states, size_t numBlocks) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBlocks) return;

    // Load as uint4 (16 bytes)
    uint4 input = reinterpret_cast<uint4*>(states)[idx];
    state_t state;
    *reinterpret_cast<uint4*>(&state) = input;

    AESEncryptDevice(&state);

    // Store back
    reinterpret_cast<uint4*>(states)[idx] = *reinterpret_cast<uint4*>(&state);
}

// Decryption
__global__ void AESDecryptKernel(state_t* states, size_t numBlocks) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBlocks) return;

    uint4 input = reinterpret_cast<uint4*>(states)[idx];
    state_t state;
    *reinterpret_cast<uint4*>(&state) = input;

    // Perform AES encryption
    AESDecryptDevice(&state);

    reinterpret_cast<uint4*>(states)[idx] = *reinterpret_cast<uint4*>(&state);
}

void h_AESEncDecECB(std::string inputFile, const std::string key, std::string outputFile, bool isDecryption) {
    std::cout << "Starting " << (isDecryption ? "Decryption" : "Encryption") << " process with input: " << inputFile << ", output: " << outputFile << std::endl;

    // Device preparation
    cudaError_t err;
    cudaSetDevice(0); // Using main GPU
    cudaFree(0);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::cerr << "Failed to get device properties.\n";
        return;
    }
    // constexpr size_t PADDED_STATE_SIZE = sizeof(state_t) + 16 - (sizeof(state_t) % 16); // Padded size for shared memory
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
    cudaEvent_t start[NUM_STREAMS], stop[NUM_STREAMS];
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
    size_t bufferSize = (1024 * 1024);
    uint8_t* buffer;
    uint8_t* d_buffer[NUM_STREAMS]; // Device buffer
    size_t bytesRead;
    // Host alloc
    cudaMallocHost(&buffer, bufferSize);
    // Cuda stream define
    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&stream[i]);
        cudaMalloc(&d_buffer[i], bufferSize); // Device buffer allocation with streams
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
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
        // size_t maxThreads = static_cast<size_t>(prop.maxThreadsPerBlock);
        size_t blockNum = bytesRead / AES_BLOCK_SIZE;
        size_t threadPblk = 256;
        size_t blocksPergrid = (blockNum + threadPblk - 1) / threadPblk;
        // size_t bp_grid = (blockNum + threadPblk - 1) / threadPblk;
        /*
        if (blockNum % num_sm > 0) threadPblk++;
        if (threadPblk > maxThreads) {
            threadPblk = maxThreads;
            num_sm = static_cast<int>(blockNum) / 1024;
            if (blockNum % 1024 > 0) {
                num_sm++;
            }
        }
        */
        // blockNum = std::min(blockNum, (static_cast<size_t>(num_sm) * 8)); // Adjust based on profile
        std::cout << "Launching kernel with " << blocksPergrid << " blocks, " << threadPblk << " threads per block\n";

        dim3 threadsPerBlock(static_cast<unsigned int>(threadPblk));
        dim3 blocksPerGrid(static_cast<unsigned int>(blocksPergrid));
        // size_t shmemSize = PADDED_STATE_SIZE * threadPblk;

        size_t starts[NUM_STREAMS], ends[NUM_STREAMS], sizes[NUM_STREAMS], nblockPerStream[NUM_STREAMS];
        size_t partSize = (blockNum + NUM_STREAMS - 1) / NUM_STREAMS;
        for (int i = 0; i < NUM_STREAMS; ++i) {
            starts[i] = i * partSize * AES_BLOCK_SIZE;
            ends[i] = std::min((i + 1) * partSize * AES_BLOCK_SIZE, bytesRead);
            sizes[i] = ends[i] - starts[i];
            nblockPerStream[i] = sizes[i] / AES_BLOCK_SIZE;

            if (sizes[i] > 0) {
                cudaMemcpyAsync(d_buffer[i], buffer + starts[i], sizes[i], cudaMemcpyHostToDevice, stream[i]);
                cudaEventRecord(start[i], stream[i]);

                if (!isDecryption) {
                    AESEncryptKernel << <blocksPerGrid, threadsPerBlock, 0, stream[i] >> > ((state_t*)(d_buffer[i]), nblockPerStream[i]);
                }
                else {
                    AESDecryptKernel << <blocksPerGrid, threadsPerBlock, 0, stream[i] >> > ((state_t*)(d_buffer[i]), nblockPerStream[i]);
                }
                if ((err = cudaGetLastError()) != cudaSuccess) {
                    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err);
                    cudaDeviceReset();
                    return;
                }
                cudaEventRecord(stop[i], stream[i]);
                cudaStreamSynchronize(stream[i]);
                cudaEventElapsedTime(&milliseconds, start[i], stop[i]);
                std::cout << "Kernel execution completed in " << milliseconds << " ms\n";

                // Accumulate the time and count the execution
                totalTime += milliseconds;
                kernelExec++;

                // Copy back to host
                cudaMemcpyAsync(buffer + starts[i], d_buffer[i], sizes[i], cudaMemcpyDeviceToHost, stream[i]);
            }
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
    // cudaFree(d_buffer);
    // cudaFree(d_roundKey);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_buffer[i]);
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
    }

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