#ifndef _SELECT_H_
#define _SELECT_H_

#include <stdint.h>

typedef struct
{
    uint32_t size;
} select_arguments_t;

typedef struct
{
    uint32_t count;
} select_results_t;

#endif