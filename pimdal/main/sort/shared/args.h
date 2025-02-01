#ifndef _KERNEL_SORT_H_
#define _KERNEL_SORT_H_

#include <stdint.h>

typedef struct
{
    uint32_t kernel_sel;
    uint32_t nr_splits;
    uint32_t nr_el;
    uint32_t offset_outer;
    uint32_t range;
    uint32_t start;
} kernel_arguments_t;

#endif