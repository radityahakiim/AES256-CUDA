#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aes_header.cuh"
#include <iostream>
#include <vector>

__device__ void ShiftRows(state_t* state) {
	int row = threadIdx.x;
	if (row > 0) {
		uint8_t a = (*state)[0][row];
		uint8_t b = (*state)[1][row];
		uint8_t c = (*state)[2][row];
		uint8_t d = (*state)[3][row];
		// Shift using registers
		if (row == 1) { 		// Second row shifts one position to left
			(*state)[0][row] = b;
			(*state)[1][row] = c;
			(*state)[2][row] = d;
			(*state)[3][row] = a;
		}
		else if (row == 2) {	// Third row shift two positions to the left
			(*state)[0][row] = c;
			(*state)[1][row] = d;
			(*state)[2][row] = a;
			(*state)[3][row] = b;
		}
		else if (row == 3) {	// Fourth row shift three positions to the left
			(*state)[0][row] = d;
			(*state)[1][row] = a;
			(*state)[2][row] = b;
			(*state)[3][row] = c;
		}
	}
}

__device__ void InvShiftRows(state_t* state) {
	int row = threadIdx.x;
	if (row > 0) {
		uint8_t a = (*state)[0][row];
		uint8_t b = (*state)[1][row];
		uint8_t c = (*state)[2][row];
		uint8_t d = (*state)[3][row];
		// Inverse shift uisng registers
		if (row == 1) {      // Second row shifts right by 1 position
			(*state)[0][row] = d;
			(*state)[1][row] = a;
			(*state)[2][row] = b;
			(*state)[3][row] = c;
		}
		else if (row == 2) { // Third row shifts right by 2 position
			(*state)[0][row] = c;
			(*state)[1][row] = d;
			(*state)[2][row] = a;
			(*state)[3][row] = b;
		}
		else if (row == 3) {// Fourth row shifts right by 3 position
			(*state)[0][row] = b;
			(*state)[1][row] = c;
			(*state)[2][row] = d;
			(*state)[3][row] = a;
		}
	}
}