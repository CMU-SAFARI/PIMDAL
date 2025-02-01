#ifndef _KERNEL_JOIN_H_
#define _KERNEL_JOIN_H_

#include <stdint.h>

typedef struct
{
    uint32_t l_count;
    int64_t date;
} query_args_t;

typedef struct
{
    uint32_t count;
} query_res_t;

#endif