#ifndef _SEL_H_
#define _SEL_H_

#include "datatype.h"

// Structures used to communicate information 
typedef struct {
    uint32_t size; // Number of elements
	uint32_t in; // Input elements in MRAM
    uint32_t out; // Elements output in MRAM
    bool (*pred)(const key_ptr_t); // Predicate function
} sel_arguments_t;

typedef struct {
    uint32_t t_count;
} sel_results_t;

/*
    Selects elements based on a predicate.
*/
int sel_kernel(sel_arguments_t *input_args, sel_results_t *result);

#endif