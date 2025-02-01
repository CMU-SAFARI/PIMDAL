#ifndef _KERNEL_JOIN_H_
#define _KERNEL_JOIN_H_

#include <stdint.h>

typedef struct
{
    uint32_t kernel_sel;
    uint32_t ptr_inner;
    uint32_t ptr_outer;
    uint32_t n_el_inner;
    uint32_t offset_outer;
    uint32_t offset_inner;
    uint32_t hash_shift;
    uint32_t n_el_outer;
} join_arguments_t;

typedef struct
{
    uint32_t count;
    uint32_t hash_shift;
} join_results_t;

#endif