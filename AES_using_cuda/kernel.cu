// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <windows.h>
#include <commdlg.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

// Function to open a file dialog for selecting an input file
inline std::string openFileDialog() {
	char filename[MAX_PATH] = { 0 };
	OPENFILENAME ofn = {};
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = NULL;
	ofn.lpstrFilter = "All Files\0*.*\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

	if (GetOpenFileName(&ofn)) {
		return std::string(filename);
	}
	return "";
}

// Function to open a save file dialog for specifying an output file
inline std::string saveFileDialog() {
	char filename[MAX_PATH] = { 0 };
	OPENFILENAME ofn = {};
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = NULL;
	ofn.lpstrFilter = "All Files\0*.*\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;

	if (GetSaveFileName(&ofn)) {
		return std::string(filename);
	}
	return "";
}

int main() {
	std::string textKey;
	char choice;
	char choice_modes;
	std::cout << "Select Modes: " << std::endl;
	std::cout << "1. CTR" << std::endl;
	std::cout << "2. ECB" << std::endl;
	std::cout << "Type here: ";
	std::cin >> choice_modes;
	if ((choice_modes != '1') && (choice_modes != '2')) {
		std::cerr << "\nPlease choose one of the modes!\n";
		return 0;
	}

	std::cout << "(E)ncrypt or (D)ecrypt? ";
	std::cin >> choice;
	if (choice != 'D' && choice != 'd') {
		if (choice != 'E' && choice != 'e') {
			std::cout << "\nInvalid input !" << std::endl;
			return 0;
		}
	}

	bool isDecryption = (choice == 'D' || choice == 'd');
	if (isDecryption)
	{
		std::cout << "Decrypt option selected";
	} else {
		std::cout << "Encrypt option selected";
	}


	std::cout << "\nEnter your encryption/decryption key (32 max): ";
	if (textKey.size() > 32) {
		std::cout << "The text key must be max. 32" << std::endl;
		return 0;
	}
	std::cin >> textKey;

	// Use file dialog to select plaintext file
	std::string inputFilePath = openFileDialog();
	if (inputFilePath.empty()) {
		std::cerr << "No input file selected." << std::endl;
		return EXIT_FAILURE;
	}

	// Use file dialog to specify the output file
	std::string outputFilePath = saveFileDialog();
	if (outputFilePath.empty()) {
		std::cerr << "No output file selected." << std::endl;
		return EXIT_FAILURE;
	}

	if (choice_modes == '1') {
		h_AESEncDecCTR(inputFilePath, textKey, outputFilePath, isDecryption);
	}
	else if (choice_modes == '2') {
		h_AESEncDecECB(inputFilePath, textKey, outputFilePath, isDecryption);
	}

	return 0;
	}
