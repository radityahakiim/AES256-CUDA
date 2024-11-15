#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>


int main() {
	std::string textKey = "passwordrahasia1234567890";
	uint8_t plaintext[] = "This is a test message for AES-256 ECB encryption! I love beer! I love Wine!! Thank you based god!!";
	uint8_t originalKey[AES_KEY_SIZE];

	size_t plaintextLength = strlen((char*)plaintext);
	size_t paddingSize = AES_BLOCK_SIZE - (plaintextLength % AES_BLOCK_SIZE);
	size_t padded_len = plaintextLength + paddingSize;

	std::vector<uint8_t> ciphertext(padded_len, 0);
	std::vector<uint8_t> paddedPlaintext(padded_len, 0);
	paddedPlaintext.resize(padded_len, paddingSize);

	// Copy plaintext and apply PKCS#7 padding
	memcpy(paddedPlaintext.data(), plaintext, plaintextLength);
	std::fill(paddedPlaintext.begin() + plaintextLength, paddedPlaintext.end(), paddingSize);

	// convert string to aes key
	convertStringToAESKey(textKey, originalKey);
	// expand the key for encryption purposes
	const size_t roundKeysSize = AES_EXPANDED_KEY_SIZE;
	uint8_t expandedKey[roundKeysSize];
	keyExpansion(expandedKey, originalKey);

	std::cout << "Padded Plaintext in hex :" << std::endl;
	for (size_t i = 0; i < padded_len; ++i) {
		printf("%02x ", paddedPlaintext[i]);
		if ((i + 1) % 16 == 0) std::cout << std::endl;
	}

	std::cout << "Original Key: " << std::endl;
	for (int i = 0; i < AES_KEY_SIZE; i++) {
		printf("%02x ", originalKey[i]);
		if ((i + 1) % 16 == 0) std::cout << std::endl;
	}

	//Print the expanded key
	std::cout << "Expanded Key: " << std::endl;
	for (int i = 0; i < AES_EXPANDED_KEY_SIZE; i++) {
		printf("%02x ", expandedKey[i]);
		if ((i + 1) % 16 == 0) std::cout << std::endl;
	}

	h_AESEncryptECB(paddedPlaintext.data(), originalKey, ciphertext.data(), padded_len);
	std::cout << "Ciphertext: " << std::endl;
	for (size_t i = 0; i < padded_len; ++i) {
		printf("%02x ", ciphertext[i]);
		if ((i + 1) % 16 == 0) std::cout << std::endl;
	}
	std::cout << std::endl;

	return 0;
}