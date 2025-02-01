#ifndef _ARITHMETIC_H_
#define _ARITHMETIC_H_

#include <stdint.h>
#include "datatype.h"

// Structures used to communicate information 
typedef struct {
    uint32_t size; // Number of input elements
	uint32_t buffer_A; // MRAM Operand 1
    uint32_t buffer_B; // MRAM Operand 2
    uint32_t buffer_out; // MRAM Output
} ar_arguments_t;

/*
    Kernel for arithmetic operations between two arrays
    Currently only supports multiplication
*/
int arithmetic_kernel(ar_arguments_t *input_args);

#endif