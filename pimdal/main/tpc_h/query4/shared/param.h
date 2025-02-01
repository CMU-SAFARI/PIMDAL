#ifndef _KERNEL_JOIN_H_
#define _KERNEL_JOIN_H_

#include <stdint.h>

typedef struct
{
    uint32_t l_count;
    uint32_t o_count;
    uint32_t date_start;
    uint32_t date_end;
    uint32_t dpu_n;
} query_args_t;

typedef struct
{
    uint32_t count;
} query_res_t;

#endif