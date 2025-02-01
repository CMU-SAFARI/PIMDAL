#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <string.h>

#define SEL_BLOCK_BYTES 1024
#define AGG_BLOCK_BYTES 1024

#ifndef PTR_TYPE
#define PTR_TYPE key_ptr32
#endif

typedef struct
{
    uint32_t key;
    uint32_t val;
    uint32_t ptr;
    uint32_t pad;
} keyval_ptr32;

typedef struct
{
    uint32_t key;
    uint32_t ptr;
} key_ptr32;

typedef struct
{
    int64_t key;
    uint32_t ptr;
    uint32_t pad;
} key_ptr64;

typedef struct
{
    char key[16];
    uint32_t ptr;
    uint32_t pad;
} key_ptrtext;

typedef struct
{
    int64_t key;
    uint32_t orderkey;
    uint32_t orderdate;
    int32_t shippriority;
} keyptr_out;

// Our main pointer datatype
typedef PTR_TYPE key_ptr_t;

inline bool empty(keyptr_out in) {
    
    return in.orderkey == 0xffffffff;
}

inline bool duplicate(keyptr_out element, keyptr_out curr) {
    bool eq = (curr.orderkey == element.orderkey) &&
              (curr.orderdate == element.orderdate) &&
              (curr.shippriority == element.shippriority);

    return eq;
}

#ifndef ENERGY
#define ENERGY 0
#endif
#define PRINT 0

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)
#endif