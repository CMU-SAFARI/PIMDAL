#ifndef _REDUCE_H_
#define _REDUCE_H_

#include "datatype.h"

// Structures used to communicate information 
typedef struct {
    uint32_t size; // Number of input elements
	uint32_t in; // MRAM Input
} red_arguments_t;

typedef struct {
    int64_t sum; // Reduction Result
} red_results_t;

int reduce_kernel(red_arguments_t *input_args, red_results_t *result);

#endif