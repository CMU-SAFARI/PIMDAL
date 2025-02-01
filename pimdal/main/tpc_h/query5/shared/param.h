#ifndef _KERNEL_JOIN_H_
#define _KERNEL_JOIN_H_

#include <stdint.h>

typedef struct
{
    uint32_t r_count;
    uint32_t n_count;
    uint32_t c_count;
    uint32_t o_count;
    uint32_t l_count;
    uint32_t s_count;
    uint32_t date_start;
    uint32_t date_end;
    char r_region[32];
    uint32_t dpu_n;
} query_args_t;

typedef struct
{
    uint32_t count;
} query_res_t;

#endif