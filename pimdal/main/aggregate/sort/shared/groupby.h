#ifndef _SELECT_H_
#define _SELECT_H_

#include <stdint.h>

typedef struct
{
    uint32_t size;
    uint32_t kernel_sel;
} groupby_arguments_t;

typedef struct
{
    uint32_t count;
} groupby_results_t;

typedef enum {
    UNIQUE=0,
    COUNT=1,
    SUM=2
} kernel_t;

#endif