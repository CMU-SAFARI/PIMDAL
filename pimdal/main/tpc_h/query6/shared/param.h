#ifndef _KERNEL_JOIN_H_
#define _KERNEL_JOIN_H_

#include <stdint.h>

typedef struct
{
    uint32_t size;
    int64_t date_start;
    int64_t date_end;
    int64_t discount;
    int64_t quantity;
} query_args_t;

typedef struct
{
    int64_t revenue;
} query_res_t;

#endif