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
    // extern __shared__ uint8_t shared_Mem[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numBlocks) return;

    state_t keystream;
    uint4* stu4 = reinterpret_cast<uint4*>(states[idx]);
    uint4 input = *stu4;

    // Prepare counter block: nonce (8 bytes) || counter (8 bytes)
    uint64_t counter_val = counterStart + idx;
    uint8_t* counterBlock = reinterpret_cast<uint8_t*>(&keystream);
    *reinterpret_cast<uint64_t*>(&counterBlock[0]) = nonce;
    *reinterpret_cast<uint64_t*>(&counterBlock[8]) = counter_val;

    // Initial rounds
    AddRoundKey(&keystream, 0, c_Rk_CTR);
    // 13 Rounds for AES-256
#pragma unroll
    for (int round = 1; round < Nr; ++round) {
        SubBytes(&keystream);
        ShiftRows(&keystream);
        MixColumns(&keystream);
        AddRoundKey(&keystream, round, c_Rk_CTR);
    }
    // Final round
    SubBytes(&keystream);
    ShiftRows(&keystream);
    AddRoundKey(&keystream, Nr, c_Rk_CTR);

    // store to destination first
    uint4* ksu4 = reinterpret_cast<uint4*>(&keystream);
    uint4 result = make_uint4(
        input.x ^ ksu4->x,
        input.y ^ ksu4->y,
        input.z ^ ksu4->z,
        input.w ^ ksu4->w
    );

    // Write back to global memory
    *stu4 = result;
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
    // cudaError_t err;
    cudaSetDevice(0); // Using main GPU
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::cerr << "Failed to get device properties.\n";
        return;
    }
    // constexpr size_t PADDED_STATE_SIZE = sizeof(state_t) + (16 - (sizeof(state_t) % 16)) % 16; // Padded size for shared memory
    int num_sm = prop.multiProcessorCount;
    std::cout << "Using a GPU with " << num_sm << " SMs\n";
    
    cudaFree(0);
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
    cudaEvent_t start[NUM_STREAMS], stop[NUM_STREAMS];
    // float milliseconds = 0;
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
    size_t bufferSize = (1024 * 1024) * 8;
    uint8_t* buffer;
    uint8_t* d_buffer[NUM_STREAMS]; // Device buffer
    size_t bytesRead;
    // Host alloc
    cudaMallocHost(&buffer, bufferSize + AES_BLOCK_SIZE);
    // Cuda stream define
    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&stream[i]);
        cudaMalloc(&d_buffer[i], bufferSize); // Device buffer allocation with streams
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
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
        size_t padBytes = ((bytesRead % AES_BLOCK_SIZE) == 0) ? 0 : (AES_BLOCK_SIZE - (bytesRead % AES_BLOCK_SIZE));
        if (padBytes > 0) memset(buffer + bytesRead, 0, padBytes);

        // Set up thread and grid
        // size_t maxThreads = static_cast<size_t>(prop.maxThreadsPerBlock);
        size_t threadPblk = 256;
        // if (blockNum % num_sm > 0) threadPblk++;
        /*
        if (threadPblk > maxThreads) {
            threadPblk = maxThreads;
            num_sm = static_cast<int>(blockNum) / 1024;
            if (blockNum % 1024 > 0) {
                num_sm++;
            }
        }
        */
        /// size_t shmemSize = sizeof(state_t) * threadPblk;
        // blockNum = std::min(blockNum, (static_cast<size_t>(num_sm) * 8)); // Adjust based on profile
        std::cout << "Launching kernel with " << num_sm << " blocks, " << threadPblk << " threads per block\n";

        dim3 threadsPerBlock(static_cast<unsigned int>(threadPblk));
        dim3 blocksPerGrid(static_cast<unsigned int>(num_sm));

        // size_t partSize = (blockNum + NUM_STREAMS - 1) / NUM_STREAMS;
        size_t sizes[NUM_STREAMS], nblockPerStream[NUM_STREAMS];
        // Partition in blocks (AES blocks), not bytes
        size_t partBlocks = (blockNum + NUM_STREAMS - 1) / NUM_STREAMS;

        for (int s = 0; s < NUM_STREAMS; ++s) {
            size_t startBlock = s * partBlocks;
            if (startBlock >= blockNum) {
                // no work for this stream
                nblockPerStream[s] = 0;
                sizes[s] = 0;
                continue;
            }
            size_t nblocks = std::min(partBlocks, blockNum - startBlock);
            nblockPerStream[s] = nblocks;
            sizes[s] = nblocks * AES_BLOCK_SIZE;           // bytes

            uint8_t* hostPtr = buffer + startBlock * AES_BLOCK_SIZE;   // correct element pointer arithmetic
            uint8_t* devPtr = reinterpret_cast<uint8_t*>(d_buffer[s]);

            // Copy host to device (element pointer ok because cudaMemcpyAsync uses bytes arg)
            cudaMemcpyAsync(devPtr, hostPtr, sizes[s], cudaMemcpyHostToDevice, stream[s]);
            cudaEventRecord(start[s], stream[s]);

            // Compute kernel grid: threads * blocks must cover nblocks
            size_t blocksNeeded = (nblocks + threadPblk - 1) / threadPblk;
            dim3 blocksPerGrid(static_cast<unsigned int>(blocksNeeded));

            // Pass counterStart as blocks (not bytes): globalCounter holds blocks processed so far
            uint64_t counterStartForStream = globalCounter + startBlock;

            AESCTRKernel << <blocksPerGrid, threadsPerBlock, 0, stream[s] >> > (
                reinterpret_cast<state_t*>(devPtr),
                nblocks,
                nonce,
                counterStartForStream
                );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Kernel launch failed on stream " << s << ": " << cudaGetErrorString(err) << "\n";
                cudaDeviceReset();
                return;
            }

            // Device to Host copy scheduled AFTER kernel completes on this stream
            cudaMemcpyAsync(hostPtr, devPtr, sizes[s], cudaMemcpyDeviceToHost, stream[s]);

            // Record stop after the copy; will indicate full roundtrip time when we synchronize
            cudaEventRecord(stop[s], stream[s]);
        }

        // Wait for all streams to finish their copies and measure time
        for (int s = 0; s < NUM_STREAMS; ++s) {
            if (nblockPerStream[s] == 0) continue;
            cudaStreamSynchronize(stream[s]); // ensure hostPtr contains final data
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start[s], stop[s]);
            totalTime += ms;
            kernelExec++;
            std::cout << "Stream " << s << " done: blocks=" << nblockPerStream[s] << ", bytes=" << sizes[s] << ", time=" << ms << " ms\n";
        }
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
    // cudaFree(d_buffer);
    // cudaFree(d_roundKey);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(stream[i]);
        cudaFree(d_buffer[i]);
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