#ifndef _PROJECT_H_
#define _PROJECT_H_

#include "datatype.h"

// Structures used to communicate information
typedef struct {
    uint32_t size; // Number of input elements
    uint32_t in; // Input array
    uint32_t out; // Output array
    key_ptr_t (*aggr)(key_ptr_t curr_val, key_ptr_t element); // Aggreagation function 
} aggr_arguments_t;

typedef struct {
    uint32_t t_count; // Number of output elements
    key_ptr_t first; // First output element
    key_ptr_t last; // Last output element
} aggr_results_t;

typedef struct{unsigned int x; unsigned int y; unsigned int z;} uint3;

/*
    Aggregate the input elements using a custom aggregation function.
*/
int group_kernel(aggr_arguments_t *input_args, aggr_results_t *result);

#endif