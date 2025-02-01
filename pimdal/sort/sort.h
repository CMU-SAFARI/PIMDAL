#ifndef _SORT_H_
#define _SORT_H_

#include "datatype.h"

// Structures used to communicate information 
typedef struct {
    uint32_t nr_elements; // Number of input elements
	uint32_t in; // Input elements in MRAM
    uint32_t out; // Elements output in MRAM
    uint32_t indices; // Partition indices in MRAM
    uint32_t nr_splits; // Number of partitions
    key_ptr_t pivot; // Initial pivot for quicksort
    key_ptr_t start; // Minimum element value
} sort_arguments_t;

/*
    Partition the input elements using quicksort partitioning.
*/
int sort_part_kernel(sort_arguments_t *input_args);

/*
    Sort the input elements using quicksort.
*/
int sort_kernel(sort_arguments_t *input_args);

#endif