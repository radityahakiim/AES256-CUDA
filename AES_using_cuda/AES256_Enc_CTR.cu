#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "aes_header.cuh"
#include <iostream>
#include <stdexcept>
#include <random>
//#include <fstream>

// __constant__ uint32_t c_Rk_CTR[AES_EXPANDED_KEY_SIZE];

const uint8_t h_sbox[256] = {
    //  00    01    02    03    04    05    06    07    08    09    0a    0b    0c    0d    0e    0f	
       0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, // 00
       0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, // 10
       0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, // 20
       0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, // 30
       0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, // 40
       0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, // 50
       0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, // 60
       0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, // 70
       0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, // 80
       0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, // 90
       0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, // a0
       0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, // b0
       0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, // c0
       0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, // d0
       0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, // e0
       0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16  // f0
};

// AddRoundKey
__device__ void AddRoundKey(state_t* state, uint8_t round, const uint32_t* roundKey) {
    size_t offset = (size_t)round * 4;
    const uint32_t* rk_ptr = roundKey + offset;
    uint32_t* state_ptr = reinterpret_cast<uint32_t*>(state);

    asm volatile(
        "{"
        ".reg .u64 s0, s1;"
        ".reg .u64 k0, k1;"
        "ld.v2.u64 {s0, s1}, [%0];"    // load 128 bits (4x32)
        "ld.v2.u64 {k0, k1}, [%1];"
        "xor.b64 s0, s0, k0;"         // two 64-bit xors instead of four 32-bit xors
        "xor.b64 s1, s1, k1;"
        "st.v2.u64 [%0], {s0, s1};"
        "}"
        :
    : "l"(state_ptr), "l"(rk_ptr)
        : "memory");
}

// MixColumns with xtime
static __device__ __forceinline__ uint32_t xtime(uint8_t x) {
    uint32_t result;
    asm volatile(
        "{\n\t"
        " .reg .u32 t;\n\t"
        " shf.r.clamp.b32 t, %1, %1, 24;\n\t"    // move msb->lsb
        " lop3.b32 %0, %1, t, 0x1B, 0xC8;\n\t"
        // truth table 0xC8 = (A << 1) ^ (B ? C : 0)
        "}\n\t"
        : "=r"(result) : "r"((uint32_t)x));
    return result;
}

__device__ __forceinline__ void MixColumns(state_t* state) {
    int lane = threadIdx.x & 3;
    uint32_t* state32 = reinterpret_cast<uint32_t*>(*state);
    uint32_t col = state32[lane];

    // Rotations by 1,2,3 bytes
    uint32_t rot1, rot2, rot3;
    asm volatile (
        "{\n\t"
        "prmt.b32 %0, %3, %3, 0x0123;\n\t"   // rot1 = rotate_left(col, 8)
        "prmt.b32 %1, %3, %3, 0x2301;\n\t"  // rot2 = rotate_left(col, 16)
        "prmt.b32 %2, %3, %3, 0x3210;\n\t"  // rot3 = rotate_left(col, 24)
        "}\n\t"
        : "=r"(rot1), "=r"(rot2), "=r"(rot3)
        : "r"(col)
        );

    uint32_t t_rep;
    asm volatile (
        "{\n\t"
        "lop3.b32 %0, %1, %2, %3, 0x96;\n\t" // rot1 ^ rot2 ^ rot3
        "xor.b32 %0, %0, %4;\n\t"           // ^ col
        "}\n\t"
        : "=r"(t_rep)
        : "r"(rot1), "r"(rot2), "r"(rot3), "r"(col)
        );

    // xtime for each byte
    uint32_t x = xtime(col);
    uint32_t x_rot1;
    asm volatile(
        "prmt.b32 %0, %1, %1, 0x0123;"
        : "=r"(x_rot1)
        : "r"(x)
        );
    uint32_t out;
    asm("{\n\t"
        "lop3.b32 %0, %1, %2, %3, 0x96;\n\t"   // out = x ^ x_rot1 ^ t_rep
        "xor.b32  %0, %0, %4;\n\t"             // out ^= col
        "}"
        : "=&r"(out)
        : "r"(x), "r"(x_rot1), "r"(t_rep), "r"(col));
    state32[lane] = out;
}

// ShiftRows
__device__ __forceinline__ void ShiftRows(state_t* state) {
    uint32_t row2, row3, row4;

    // Load each row (4 bytes) into a 32-bit register
    row2 = *((uint32_t*)&((*state)[1][0]));
    row3 = *((uint32_t*)&((*state)[2][0]));
    row4 = *((uint32_t*)&((*state)[3][0]));

    // Use PTX inline assembly with barrel shift (shf.l.wrap.b32)
    asm volatile (
        // row2: rotate left by 1 byte
        "prmt.b32 %0, %0, %0, 0x041302;\n\t"
        // row3: rotate left by 2 bytes
        "prmt.b32 %1, %1, %1, 0x2301;\n\t"
        // row4: rotate left by 3 bytes
        "prmt.b32 %2, %2, %2, 0x3201;\n\t"
        : "+r"(row2), "+r"(row3), "+r"(row4)
        );

    // Store back to the state
    *((uint32_t*)&((*state)[1][0])) = row2;
    *((uint32_t*)&((*state)[2][0])) = row3;
    *((uint32_t*)&((*state)[3][0])) = row4;
}

// SubBytes
__device__ __forceinline__ void SubBytes(state_t* state, uint8_t* sbox) {
    int idx = threadIdx.x;
    if (idx < 16) {
        int row = idx & 0x3;
        int col = idx >> 2;
        (*state)[row][col] = d_sb[(*state)[row][col]];
    }
}

// Encryption
__global__ void AESCTRKernel(state_t* states, size_t numBlocks, uint64_t nonce, uint64_t counterStart, const uint8_t* dev_sbox, const uint32_t* roundKey) {
    // extern __shared__ uint8_t shared_Mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks) return;

    __shared__ uint8_t shared_sbox[256];
    __shared__ uint32_t shared_rk[AES_EXPANDED_KEY_SIZE];
    int tid = threadIdx.x;
    if (tid < 256)
        shared_sbox[tid] = dev_sbox[tid];
    __syncthreads();

    if (tid < AES_EXPANDED_KEY_SIZE)
        shared_rk[tid] = roundKey[tid];
    __syncthreads();

    state_t keystream;

    // Load input
    ulonglong2 input128 = *reinterpret_cast<ulonglong2*>(&states[idx]);

    // Prepare counter block: nonce (8 bytes) || counter (8 bytes)
    uint64_t counter_val = counterStart + idx;
    // uint8_t* counterBlock = reinterpret_cast<uint8_t*>(&keystream);
    ((uint64_t*)&keystream)[0] = nonce;
    ((uint64_t*)&keystream)[1] = counter_val;

    // Initial rounds
    AddRoundKey(&keystream, 0, shared_rk);
    // 13 Rounds for AES-256
#pragma unroll
    for (int round = 1; round < Nr; ++round) {
        SubBytes(&keystream, shared_sbox);
        ShiftRows(&keystream);
        MixColumns(&keystream);
        AddRoundKey(&keystream, round, shared_rk);
    }
    // Final round
    SubBytes(&keystream, shared_sbox);
    ShiftRows(&keystream);
    AddRoundKey(&keystream, Nr, shared_rk);

    // XOR in 64-bit chunks instead of 4x32
    ulonglong2* ks128 = reinterpret_cast<ulonglong2*>(&keystream);
    ulonglong2 result;
    result.x = input128.x ^ ks128->x;
    result.y = input128.y ^ ks128->y;

    // Write back
    *reinterpret_cast<ulonglong2*>(states[idx]) = result;
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
    uint32_t* d_roundKey;
    cudaMalloc(&d_roundKey, AES_EXPANDED_KEY_SIZE);
    cudaMemcpy(d_roundKey, expandedKey, AES_EXPANDED_KEY_SIZE, cudaMemcpyHostToDevice);
    cudaFreeHost(expandedKey);
    std::cout << "Expanded key copied to constant\n";

    // S-box initialization
    // SBoxInit(false);
    uint8_t* d_sbox;
    cudaMalloc(&d_sbox, sizeof(h_sbox));
    cudaMemcpy(d_sbox, h_sbox, sizeof(h_sbox), cudaMemcpyHostToDevice);
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
    size_t bufferSize = (1024 * 1024) * 32;
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

    uint64_t globalCounter = 0;

    if (isDecryption) {
        fseek(file_in, inputOffset, SEEK_SET);
    }

    size_t blocksNeeded;

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
        std::cout << "Launching kernel with " << threadPblk << " threads per block\n";

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
            blocksNeeded = (nblocks + threadPblk - 1) / threadPblk;
            dim3 blocksPerGrid(static_cast<unsigned int>(blocksNeeded));

            // Pass counterStart as blocks (not bytes): globalCounter holds blocks processed so far
            uint64_t counterStartForStream = globalCounter + startBlock;

            AESCTRKernel << <blocksPerGrid, threadsPerBlock, 0, stream[s] >> > (
                reinterpret_cast<state_t*>(devPtr),
                nblocks,
                nonce,
                counterStartForStream,
                d_sbox,
                d_roundKey
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
            std::cout << "Grids : " << blocksNeeded << std::endl;
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
    cudaFree(d_sbox);
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