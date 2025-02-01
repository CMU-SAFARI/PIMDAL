#ifndef _SORTMERGE_H_
#define _SORTMERGE_H_

#include <stdint.h>

typedef struct
{
    uint32_t inner; // Sorted inner relation in MRAM
    uint32_t size_inner; // Inner relation number of elements
    uint32_t outer; // Sorted outer relation in MRAM
    uint32_t size_outer; // Outer relation number of elements
    uint32_t out; // Joined elements output in MRAM
} merge_arguments_t;

typedef struct
{
    uint32_t t_count; // Number of output elements
} merge_results_t;

/*
    Join to sorted relations.
*/
int merge_kernel(merge_arguments_t *input_args, merge_results_t *result);

#endif