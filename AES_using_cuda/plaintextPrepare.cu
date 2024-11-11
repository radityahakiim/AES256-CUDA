#include <iostream>
#include <vector>
#include <cstring>
#include "aes_header.cuh"

std::vector<uint8_t> preparePlaintext(const std::string& input) {
	const int blockSize = AES_BLOCK_SIZE;
	size_t inputLength = input.size();

	size_t paddedLength = ((inputLength / blockSize) + 1) * blockSize;

	std::vector<uint8_t> paddedPlaintext(paddedLength);

	std::memcpy(paddedPlaintext.data(), input.data(), inputLength);

	uint8_t paddingValue = blockSize - (inputLength % blockSize);
	for (size_t i = inputLength; i < paddedLength; ++i) {
		paddedPlaintext[i] = paddingValue;
	}
	return paddedPlaintext;
}

void printHex(const std::vector<uint8_t>& data) {
	for (auto byte : data) {
		printf("%02x ", byte);
	}
	std::cout << std::endl;
}